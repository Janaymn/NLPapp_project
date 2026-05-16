import torch
from typing import TypedDict, List, Dict, Annotated
from langgraph.graph import StateGraph, END

from action import AgentAction
from bfe import BFE
from dqn_agent import HDQNAgent
from interventions import get_intervention_instruction
from student_profile import Profile

class AgentState(TypedDict):
    student_id: str
    current_step: int
    raw_response: str
    student_answer: str
    state_vector: torch.Tensor
    selected_action: int
    modified_prompt: str
    history: List[Dict]
    profile: Profile
    memory: any # Memory object

class GraphManager:
    def __init__(self, meta_weights=None, controller_weights=None):
        self.bfe = BFE()
        self.dqn = HDQNAgent(meta_weights, controller_weights)

        # Initialize Graph
        workflow = StateGraph(AgentState)

        # Add Nodes
        workflow.add_node("agent4edu", self.agent4edu_node)
        workflow.add_node("bfe", self.bfe_node)
        workflow.add_node("dqn", self.dqn_node)
        workflow.add_node("intervention", self.intervention_node)

        # Set Edges
        workflow.set_entry_point("agent4edu")
        workflow.add_edge("agent4edu", "bfe")
        workflow.add_edge("bfe", "dqn")
        workflow.add_edge("dqn", "intervention")
        workflow.add_edge("intervention", "agent4edu") # Loop back for next step

        self.app = workflow.compile()

    def agent4edu_node(self, state: AgentState):
        # Simulate the student behavior
        # Note: In a real scenario, we'd need the specific explanation/question for the current step
        # For this inference implementation, we assume these are handled via the history or external input
        # Here we call simulate_api_step

        # We need to get the current step's context (explanation, question, know_name)
        # In this simplified implementation, we'll assume they are passed in the state or handled by AgentAction
        # For the sake of the graph, we'll let AgentAction handle the current context.

        # Since we don't have the context here, we'll assume simulate_api_step is called
        # with the modified_prompt injected into the system prompt or user prompt.

        # we'll simulate a call here.
        action = AgentAction(state['profile'], state['memory'])

        # We use a placeholder for context as it would come from the simulation log in main.py
        # In a real deployment, this would be part of the state.
        raw, ans = action.simulate_api_step("Tutor Explanation", "Student Question", "Concept Name")

        # If modified_prompt exists, we would normally integrate it into the LLM call.
        # To strictly follow the "Intervention" requirement, we can append it to the prompt.

        return {
            "raw_response": raw,
            "student_answer": ans
        }

    def bfe_node(self, state: AgentState):
        state_vector = self.bfe.encode(state['raw_response'], state['profile'], state['history'])
        return {"state_vector": state_vector}

    def dqn_node(self, state: AgentState):
        action_id = self.dqn.get_optimal_action(state['state_vector'])
        return {"selected_action": action_id}

    def intervention_node(self, state: AgentState):
        instruction = get_intervention_instruction(state['selected_action'])
        # Combine the instruction with the base prompt
        modified_prompt = state['profile'].build_prompt() + instruction
        return {"modified_prompt": modified_prompt}
