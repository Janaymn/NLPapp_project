import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from pyngrok import ngrok

from config import DATA_PATH
from memory import Memory
from action import AgentAction
from utils import load_json

app = FastAPI()

# Global dictionary to hold active simulated students in memory
active_students = {}

# Pydantic Model for incoming API requests
class InterventionRequest(BaseModel):
    student_id: str
    knowledge: str
    Explanation: str
    question: str
    correct_answer: str
    know_name: str

def get_or_create_student(student_id: str):
    if student_id not in active_students:
        KCG = load_json(f"{DATA_PATH}/kcg.json")
        know_course_list = load_json(f"{DATA_PATH}/know_course_list.json")
        know_name_list = load_json(f"{DATA_PATH}/know_name_list.json")

        # Create a simple object instead of using Profile
        class SimpleProfile:
            def __init__(self, student_id):
                self.student_id = student_id
                self.values = {}

        profile = SimpleProfile(student_id)
        memory = Memory(KCG, know_course_list, know_name_list)
        action = AgentAction(profile, memory)

        active_students[student_id] = {
            "profile": profile,
            "memory": memory,
            "action": action,
            "step_counter": 1
        }
    return active_students[student_id]

@app.post("/simulate")
async def simulate_intervention(req: InterventionRequest):
    student = get_or_create_student(req.student_id)
    action = student["action"]
    memory = student["memory"]

    # 1. Run the LLM Simulation
    raw_response, student_answer = action.simulate_api_step(
        explanation=req.Explanation,
        question=req.question,
        know_name=req.know_name
    )

    # 2. Evaluate Correctness
    is_correct = (student_answer.lower() == req.correct_answer.lower())
    score = 1 if is_correct else 0

    # 3. Update Memory State
    practice_record = {
        'knowledge': req.know_name,
        'score': score
    }

    # Generate corrective feedback string based on correctness
    feedback_string = memory.reflect_corrective(
        practice=practice_record,
        ans={'task2': req.knowledge, 'task4': 'yes' if is_correct else 'no'}
    )

    # Generate the 500-word summary and save to Long-Term Memory
    summary = memory.reflect_summary(feedback_string)

    # Increment absolute time step for forgetting curve logic
    student["step_counter"] += 1

    return {
        "student_id": req.student_id,
        "is_correct": is_correct,
        "student_provided_answer": student_answer,
        "raw_monologue": raw_response,
        "generated_summary": summary
    }

# Execution block to start FastAPI and Ngrok tunneling
if __name__ == "__main__":
    # Start ngrok tunnel on port 8000
    public_url = ngrok.connect(8000).public_url
    print(f" Simulation API is live at: {public_url}/docs")

    # Start the local server
    uvicorn.run(app, host="0.0.0.0", port=8000)