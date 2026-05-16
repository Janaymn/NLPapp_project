"""
Mapping of Hierarchical DQN actions to prompt rewriting strategies.
"""

META_STRATEGIES = {
    0: {"name": "Simplification", "desc": "Reduce cognitive load by simplifying language."},
    1: {"name": "Scaffolding", "desc": "Break down complex concepts into steps."},
    2: {"name": "Analogy", "desc": "Use real-world comparisons."},
    3: {"name": "Motivation", "desc": "Encourage persistence and growth mindset."},
    4: {"name": "Concrete Example", "desc": "Provide specific, tangible examples."},
    5: {"name": "Contrastive Explanation", "desc": "Contrast a correct concept with a common misconception."},
    6: {"name": "Socratic Questioning", "desc": "Guide the student using strategic questions."},
    7: {"name": "Visual Description", "desc": "Describe the concept using spatial or visual cues."},
    8: {"name": "Summary Recap", "desc": "Summarize key points before proceeding."}
}

SUB_STRATEGIES = {
    0: {"name": "Subtle", "instruction": "Integrate this approach naturally into the flow without being overly explicit."},
    1: {"name": "Standard", "instruction": "Apply this strategy clearly and explicitly to guide the student."},
    2: {"name": "Intensive", "instruction": "Prioritize this strategy heavily; restructure the response to center around this approach."}
}

def get_intervention_instruction(goal, action):
    """
    Retrieve the hierarchical prompt instruction for a given goal and action.
    """
    meta = META_STRATEGIES.get(goal, META_STRATEGIES[0])
    sub = SUB_STRATEGIES.get(action, SUB_STRATEGIES[0])

    # Define specific core instructions for each meta-strategy
    core_instructions = {
        0: "Rewrite the explanation using common, everyday vocabulary. Avoid academic jargon and complex terminology.",
        1: "Break the explanation into a clear, numbered list of 3-4 simple points to guide the student through the logic.",
        2: "Explain the concept using a relatable, real-world analogy that connects the abstract concept to daily life.",
        3: "Start the response by encouraging the student and acknowledging that making mistakes is part of learning.",
        4: "Provide a specific, concrete example that illustrates the concept in a practical scenario.",
        5: "Contrast the correct reasoning with a common misconception to clarify the concept.",
        6: "Instead of giving the answer immediately, ask a guiding question that helps the student discover the solution.",
        7: "Describe the concept using visual or spatial metaphors to help the student visualize the process.",
        8: "Start by summarizing the key points of the concept before explaining the specific problem."
    }

    core_text = core_instructions.get(goal, core_instructions[0])
    style_text = sub["instruction"]

    return f"\n\n[INTERVENTION: {meta['name']} - {sub['name']}]\n{style_text}\n{core_text}"
