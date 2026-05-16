import torch
import torch.nn as nn
import numpy as np
import os
import asyncio
from collections import deque
from tqdm import tqdm
from config import DATA_PATH, RL_PARAMS
from utils import load_json, save_json
from dqn_agent import HDQNAgent
from trainer import DQNTrainer, ReplayBuffer
from student_profile import Profile
from memory import Memory
from graph_manager import GraphManager
from action import AgentAction
from interventions import get_intervention_instruction

def evaluate_performance(agent, buffer, batch_size):
    """
    Evaluate the pedagogical performance of the agent.
    Returns a dictionary of metrics: k, engagement, efficiency, reward, and stability.
    """
    if len(buffer) < batch_size:
        return None

    states, goals, actions, rewards, reward_breakdown, next_states, dones = buffer.sample(batch_size)

    # 1. Calculate average reward components
    avg_k = torch.mean(reward_breakdown[:, 0]).item()
    avg_eng = torch.mean(reward_breakdown[:, 1]).item()
    avg_eff = torch.mean(reward_breakdown[:, 2]).item()
    avg_reward = torch.mean(rewards).item()

    # 2. Stability (Policy Consistency)
    correct = 0
    with torch.no_grad():
        # Meta-controller consistency
        meta_q = agent.meta_controller(states)
        meta_target_q = agent.meta_target(states)
        meta_correct = (torch.argmax(meta_q, dim=1) == torch.argmax(meta_target_q, dim=1)).sum().item()

        # Controller consistency
        goal_tensor = goals.unsqueeze(1)
        augmented_state = torch.cat([states, goal_tensor], dim=1)
        ctrl_q = agent.controller(augmented_state)
        ctrl_target_q = agent.controller_target(augmented_state)
        ctrl_correct = (torch.argmax(ctrl_q, dim=1) == torch.argmax(ctrl_target_q, dim=1)).sum().item()

    stability = (meta_correct + ctrl_correct) / (2 * batch_size)

    return {
        "k": avg_k,
        "engagement": avg_eng,
        "efficiency": avg_eff,
        "reward": avg_reward,
        "stability": stability
    }

async def train_agent(agent, trainer, buffer, gm, train_logs, KCG, know_course_list, know_name, num_episodes=50):
    """
    Training phase: learns from the training dataset.
    """
    from neurodiverse_profiles import NEURODIVERSE_PROFILES

    eval_interval = 100  # Evaluate every 100 steps
    checkpoint_interval = 500  # Save every 500 steps

    # Determine total_steps based on loaded agent
    # We'll look for the highest checkpoint step
    checkpoint_files = [f for f in os.listdir("checkpoints") if f.startswith("meta_controller_") and f.endswith(".pth")]
    if checkpoint_files:
        # Extract the step number specifically from 'meta_controller_{step}.pth'
        total_steps = max([int(f.replace("meta_controller_", "").replace(".pth", "")) for f in checkpoint_files])
        print(f"Resuming training from step {total_steps}")
    else:
        total_steps = 0

    # Progress bar for episodes
    ep_pbar = tqdm(range(num_episodes), desc="Training Episodes")

    for episode in ep_pbar:
        student_idx = np.random.randint(0, len(train_logs))
        logs = train_logs[student_idx]['logs']
        student_id = train_logs[student_idx]['user_id']

        profile = Profile(student_id, profile_data=NEURODIVERSE_PROFILES["ADHD_DYSLEXIA"])
        memory = Memory(KCG, know_course_list, know_name)
        action_sim = AgentAction(profile, memory)

        history = []
        episode_reward = 0

        # Progress bar for steps within the episode
        step_pbar = tqdm(enumerate(logs), total=len(logs), desc=f"Ep {episode}", leave=False)

        for rec_id, rec in step_pbar:
            raw_response = rec.get('exer_analysis', rec.get('exer_content', ''))
            state_vector = gm.bfe.encode(raw_response, profile, history, knowledge_code=rec.get('knowledge_code'), kcg=KCG)

            epsilon = max(RL_PARAMS['EPSILON_END'], RL_PARAMS['EPSILON_START'] * (RL_PARAMS['EPSILON_DECAY'] ** total_steps))

            if np.random.random() < epsilon:
                goal = np.random.randint(0, 9)
                action = np.random.randint(0, 3)
            else:
                with torch.no_grad():
                    goal_q = agent.meta_controller(state_vector)
                    goal = torch.argmax(goal_q).item()
                    goal_tensor = torch.FloatTensor([goal])
                    augmented_state = torch.cat([state_vector, goal_tensor])
                    action_q = agent.controller(augmented_state)
                    action = torch.argmax(action_q).item()

            intervention_text = get_intervention_instruction(goal, action)
            modified_prompt = profile.build_prompt() + intervention_text
            explanation = rec.get('exer_content', 'Tutor Explanation')
            question = rec.get('exer_content', 'Student Question')
            know_name_val = rec.get('know_name', 'Concept')

            sim_raw, sim_ans = action_sim.simulate_api_step(
                explanation, question, know_name_val, modified_prompt=modified_prompt
            )

            ground_truth = rec.get('exer_answer', '').lower().strip()
            score = 1.0 if (ground_truth in sim_ans.lower()) else 0.0
            confusion = gm.bfe._analyze_confusion(sim_raw)
            length = min(1.0, len(sim_raw) / 200.0) if sim_raw else 0.0

            k, engagement, time_efficiency = score, 1.0 - confusion, 1.0 - length
            weights = RL_PARAMS['REWARD_WEIGHTS']
            reward = (weights['w1'] * k) + (weights['w2'] * engagement) + (weights['w3'] * time_efficiency)
            reward = float(reward)

            if rec_id + 1 < len(logs):
                next_rec = logs[rec_id + 1]
                next_raw = next_rec.get('exer_analysis', next_rec.get('exer_content', ''))
                next_state_vector = gm.bfe.encode(
                    next_raw,
                    profile,
                    history + [{'raw': sim_raw, 'ans': sim_ans, 'corr': score}],
                    knowledge_code=next_rec.get('knowledge_code'),
                    kcg=KCG
                )
            else:
                next_state_vector = state_vector

            done = 1.0 if rec_id + 1 == len(logs) else 0.0
            s = torch.FloatTensor(state_vector)
            ns = torch.FloatTensor(next_state_vector)
            reward_breakdown = torch.FloatTensor([k, engagement, time_efficiency])
            buffer.push(s, goal, action, reward, reward_breakdown, ns, done)

            meta_loss, ctrl_loss = trainer.train_step(buffer)
            history.append({'raw': sim_raw, 'ans': sim_ans, 'corr': score})
            total_steps += 1
            episode_reward += reward

            # Print evaluations ALL THE TIME (per step) instead of just every 100 steps
            metrics = evaluate_performance(agent, buffer, RL_PARAMS['BATCH_SIZE'])
            if metrics:
                step_pbar.set_postfix({
                    "k": f"{metrics['k']:.2f}",
                    "Eng": f"{metrics['engagement']:.2f}",
                    "Rew": f"{metrics['reward']:.2f}",
                    "Stab": f"{metrics['stability']:.2f}"
                })

            if total_steps % RL_PARAMS['TARGET_UPDATE_FREQ'] == 0:
                trainer.update_target_networks()

            if total_steps % checkpoint_interval == 0:
                torch.save(agent.meta_controller.state_dict(), f"checkpoints/meta_controller_{total_steps}.pth")
                torch.save(agent.controller.state_dict(), f"checkpoints/controller_{total_steps}.pth")

        ep_pbar.set_postfix({"LastEpReward": f"{episode_reward:.2f}"})

async def test_agent(agent, test_logs, KCG, know_course_list, know_name):
    """
    Testing phase: evaluates the trained agent on unseen students.
    """
    from neurodiverse_profiles import NEURODIVERSE_PROFILES
    print("\nStarting Evaluation on Test Set...")

    total_rewards = []
    total_ks = []

    # Evaluate on a subset of the test set for efficiency
    test_subset = test_logs[:10]

    for student in tqdm(test_subset, desc="Testing Students"):
        logs = student['logs']
        student_id = student['user_id']
        profile = Profile(student_id, profile_data=NEURODIVERSE_PROFILES["ADHD_DYSLEXIA"])
        memory = Memory(KCG, know_course_list, know_name)
        action_sim = AgentAction(profile, memory)

        history = []
        episode_reward = 0
        episode_k = 0

        from bfe import BFE
        bfe_tester = BFE()

        for rec in logs:
            raw_response = rec.get('exer_analysis', rec.get('exer_content', ''))
            state_vector = bfe_tester.encode(raw_response, profile, history, knowledge_code=rec.get('knowledge_code'), kcg=KCG)

            with torch.no_grad():
                goal_q = agent.meta_controller(state_vector)
                goal = torch.argmax(goal_q).item()
                goal_tensor = torch.FloatTensor([goal])
                augmented_state = torch.cat([state_vector, goal_tensor])
                action_q = agent.controller(augmented_state)
                action = torch.argmax(action_q).item()

            intervention_text = get_intervention_instruction(goal, action)
            modified_prompt = profile.build_prompt() + intervention_text

            sim_raw, sim_ans = action_sim.simulate_api_step(
                rec.get('exer_content', 'Tutor Explanation'),
                rec.get('exer_content', 'Student Question'),
                rec.get('know_name', 'Concept'),
                modified_prompt=modified_prompt
            )

            ground_truth = rec.get('exer_answer', '').lower().strip()
            score = 1.0 if (ground_truth in sim_ans.lower()) else 0.0

            confusion = bfe_tester._analyze_confusion(sim_raw)
            length = min(1.0, len(sim_raw) / 200.0) if sim_raw else 0.0
            weights = RL_PARAMS['REWARD_WEIGHTS']
            reward = (weights['w1'] * score) + (weights['w2'] * (1-confusion)) + (weights['w3'] * (1-length))

            episode_reward += reward
            episode_k += score
            history.append({'raw': sim_raw, 'ans': sim_ans, 'corr': score})

        total_rewards.append(episode_reward)
        total_ks.append(episode_k / max(len(logs), 1))

    print(f"\nTest Results:")
    print(f"Average Episode Reward: {np.mean(total_rewards):.3f}")
    print(f"Average Knowledge Gain (k): {np.mean(total_ks):.3f}")

async def main():
    print("Initializing Pipeline...")
    os.makedirs("checkpoints", exist_ok=True)

    # Data Setup
    all_logs = load_json(f"{DATA_PATH}/stu_logs.json")
    KCG = load_json(f"{DATA_PATH}/kcg.json")
    know_course_list = load_json(f"{DATA_PATH}/know_course_list.json")
    know_name = load_json(f"{DATA_PATH}/know_name_list.json")

    # Split Data: 80% Train, 20% Test
    np.random.shuffle(all_logs)
    if len(all_logs) < 2:
        train_logs = all_logs
        test_logs = all_logs
    else:
        split_idx = int(len(all_logs) * 0.8)
        train_logs = all_logs[:split_idx]
        test_logs = all_logs[split_idx:]

    print(f"Data split: {len(train_logs)} training students, {len(test_logs)} testing students.")

    # Initialize Agent and Trainer
    agent = HDQNAgent()
    trainer = DQNTrainer(agent)
    buffer = ReplayBuffer()
    gm = GraphManager()
    gm.dqn = agent

    # 1. Training Phase
    await train_agent(agent, trainer, buffer, gm, train_logs, KCG, know_course_list, know_name)

    # 2. Testing Phase
    await test_agent(agent, test_logs, KCG, know_course_list, know_name)

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
