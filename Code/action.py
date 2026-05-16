import re
from llm_client import LLMClient

class AgentAction:
    def __init__(self, profile, memory):
        self.profile = profile
        self.memory = memory
        self.llm = LLMClient()

    def simulate_api_step(self, explanation, question, know_name, modified_prompt=None):
        short_mem = self.memory.retrieve_short()
        long_mem = self.memory.retrieve_long()

        # Build the dynamic, comprehension-aware prompt
        prompt = (
            "You are taking an interactive lesson.\n\n"
            f"Knowledge Concept: {know_name}\n"
            f"Tutor's Explanation: {explanation}\n"
            f"Question: {question}\n\n"
        )

        prompt += (
            "CRITICAL INSTRUCTION ON COMPREHENSION:\n"
            "You must evaluate the 'Tutor's Explanation' based on your # Profile # and your Memory.\n"
            "1. If the explanation uses complex terminology, academic jargon, or topics outside of your practiced knowledge, "
            "you MUST simulate confusion, narrate your misunderstanding, and you will likely answer the question incorrectly.\n"
            "2. If the explanation is simplified, conversational, or aligns with your profile and memories, "
            "you will understand it and have a high chance of reasoning correctly.\n\n"
        )

        # Inject Memory States
        learning_status = long_mem.get('learning_status', [])
        status_summary = learning_status[-1] if learning_status else 'None'
        prompt += f"Your current Learning Status Summary: {status_summary}\n\n"

        prompt += (
            "Task 1: Provide your internal monologue. Analyze the explanation. Do you understand the words used based on your ability? "
            "How does this connect to your past knowledge?\n"
            "Task 2: State your final predicted answer strictly wrapped in brackets, e.g., [Final Answer: X].\n"
        )

        # Use modified_prompt if provided, otherwise build default
        system_content = modified_prompt if modified_prompt else self.profile.build_prompt()

        messages = [
            {'role': 'system', 'content': system_content},
            {'role': 'user', 'content': prompt}
        ]

        # Call Groq
        raw_response = self.llm.call(messages)
        
        # Extract the final answer using regex based on the requested format
        match = re.search(r'\[Final Answer:\s*(.*?)\]', raw_response, re.IGNORECASE)
        student_answer = match.group(1).strip() if match else "Unknown"

        return raw_response, student_answer