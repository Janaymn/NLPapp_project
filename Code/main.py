import asyncio
from config import DATA_PATH, RESULT_PATH
from student_profile import Profile
from memory import Memory
from action import AgentAction
from utils import load_json, save_json

async def run_for_student(student_index):
    print(student_index)
    all_logs = load_json(f"{DATA_PATH}/stu_logs.json")
    logs = all_logs[student_index]['logs']
    student_id = all_logs[student_index]['user_id']
    KCG = load_json(f"{DATA_PATH}/kcg.json")
    know_course_list = load_json(f"{DATA_PATH}/know_course_list.json")
    know_name = load_json(f"{DATA_PATH}/know_name_list.json")

    # Integration: Use Neurodiverse Profile (ADHD/Dyslexia)
    from neurodiverse_profiles import NEURODIVERSE_PROFILES
    profile = Profile(student_id, profile_data=NEURODIVERSE_PROFILES["ADHD_DYSLEXIA"])

    memory = Memory(KCG, know_course_list, know_name)

    # Initialize DQN Graph Manager
    from graph_manager import GraphManager
    gm = GraphManager()

    results = []

    # State for LangGraph
    state = {
        "student_id": student_id,
        "current_step": 0,
        "raw_response": "",
        "student_answer": "",
        "state_vector": None,
        "selected_action": 0,
        "modified_prompt": profile.build_prompt(),
        "history": [],
        "profile": profile,
        "memory": memory
    }

    for rec_id in range(len(logs)):
        rec = logs[rec_id]
        state["current_step"] = rec_id + 1

        # Run one iteration of the graph
        # Since LangGraph usually runs until END, we can manually invoke nodes
        # or use a custom loop. For inference, we'll manually invoke the sequence.

        # 1. Agent4Edu Node
        from action import AgentAction
        action = AgentAction(profile, memory)
        # We pass the actual log data (rec) to simulate the step
        raw, ans = action.simulate_api_step(
            rec['explanation'],
            rec['question'],
            rec['know_name'],
            modified_prompt=state["modified_prompt"]
        )
        state["raw_response"] = raw
        state["student_answer"] = ans

        # 2. BFE Node
        state["state_vector"] = gm.bfe.encode(raw, profile, state["history"])

        # 3. DQN Node
        state["selected_action"] = gm.dqn.get_optimal_action(state["state_vector"])

        # 4. Intervention Node
        from interventions import get_intervention_instruction
        instruction = get_intervention_instruction(state["selected_action"])
        state["modified_prompt"] = profile.build_prompt() + instruction

        # Store results
        results.append({
            'ans': ans,
            'raw': raw,
            'action': state["selected_action"],
            'prompt': state["modified_prompt"]
        })

        # Update history
        state["history"].append({'raw': raw, 'ans': ans, 'corr': 1 if ans == rec['answer'] else 0})

    save_json(f"{RESULT_PATH}/{student_id}_results.json", results)

async def main():
    students = range(1)
    # user_idx_start = 0
    # user_idx_end = 1
    # agent_id_list = load_json(f"{DATA_PATH}/agent_id_list.json")
    # students = agent_id_list[user_idx_start:user_idx_end]
    
    await asyncio.gather(*[run_for_student(s) for s in students])

if __name__ == '__main__':
    asyncio.run(main())
