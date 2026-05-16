import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from config import RL_PARAMS

class ReplayBuffer:
    """
    Experience Replay Buffer to store transitions for HDQN.
    """
    def __init__(self, capacity=RL_PARAMS['MEMORY_SIZE']):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, goal, action, reward, reward_breakdown, next_state, done):
        self.buffer.append((state, goal, action, reward, reward_breakdown, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, goal, action, reward, reward_breakdown, next_state, done = zip(*batch)

        return (
            torch.FloatTensor(torch.stack(state)),
            torch.tensor(goal, dtype=torch.float32),
            torch.LongTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(torch.stack(reward_breakdown)),
            torch.FloatTensor(torch.stack(next_state)),
            torch.FloatTensor(done)
        )

    def __len__(self):
        return len(self.buffer)

class DQNTrainer:
    """
    Trainer for HDQN, optimizing both Meta-Controller and Controller.
    """
    def __init__(self, agent):
        self.agent = agent
        self.gamma = RL_PARAMS['GAMMA']
        self.batch_size = RL_PARAMS['BATCH_SIZE']

        # Optimizers
        self.meta_optimizer = optim.Adam(self.agent.meta_controller.parameters(), lr=RL_PARAMS['LR'])
        self.controller_optimizer = optim.Adam(self.agent.controller.parameters(), lr=RL_PARAMS['LR'])

        self.criterion = nn.MSELoss()

    def train_step(self, buffer):
        if len(buffer) < self.batch_size:
            return 0.0, 0.0

        # Sample transition
        states, goals, actions, rewards, reward_breakdown, next_states, dones = buffer.sample(self.batch_size)

        # --- 1. Train Meta-Controller (Double DQN) ---
        with torch.no_grad():
            # DDQN: use online network to pick best action, target network to evaluate it
            best_next_meta_action = self.agent.meta_controller(next_states).argmax(1).unsqueeze(1)
            next_meta_q = self.agent.meta_target(next_states).gather(1, best_next_meta_action).squeeze()
            meta_targets = rewards + (1 - dones) * self.gamma * next_meta_q

        # Current Q-value for the goal that was chosen
        goal_indices = torch.tensor([[g] for g in goals]).flatten().unsqueeze(1).long()
        current_meta_q = self.agent.meta_controller(states).gather(1, goal_indices).squeeze()

        meta_loss = self.criterion(current_meta_q, meta_targets)

        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

        # --- 2. Train Controller (Double DQN + Dynamic Weighting η) ---
        goal_tensor = goals.unsqueeze(1)
        current_augmented_state = torch.cat([states, goal_tensor], dim=1)

        with torch.no_grad():
            # Calculate Dynamic Weighting Factor η = Q_high(s_{t+1}, i) / Q_high(s_t, i_t)
            q_high_next = self.agent.meta_controller(next_states).gather(1, goal_indices).squeeze()
            q_high_curr = self.agent.meta_controller(states).gather(1, goal_indices).squeeze()
            # Avoid division by zero
            eta = q_high_next / (q_high_curr + 1e-5)
            eta = torch.clamp(eta, 0.1, 2.0) # Heuristic clamping

            # DDQN: use online network to pick best action
            next_augmented_state = torch.cat([next_states, goal_tensor], dim=1)
            best_next_ctrl_action = self.agent.controller(next_augmented_state).argmax(1).unsqueeze(1)
            next_ctrl_q = self.agent.controller_target(next_augmented_state).gather(1, best_next_ctrl_action).squeeze()

            # Use η to weight the future reward
            ctrl_targets = rewards + (1 - dones) * self.gamma * eta * next_ctrl_q

        current_ctrl_q = self.agent.controller(current_augmented_state).gather(1, actions.unsqueeze(1)).squeeze()
        ctrl_loss = self.criterion(current_ctrl_q, ctrl_targets)

        self.controller_optimizer.zero_grad()
        ctrl_loss.backward()
        self.controller_optimizer.step()

        return meta_loss.item(), ctrl_loss.item()

    def update_target_networks(self):
        """
        Copy weights from main networks to target networks.
        """
        self.agent.meta_target.load_state_dict(self.agent.meta_controller.state_dict())
        self.agent.controller_target.load_state_dict(self.agent.controller.state_dict())
