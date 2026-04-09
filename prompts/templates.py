def standard_prompt(question):
    """Condition 1: Direct question, no special prompting."""
    return f"""Solve the following math problem. Give only the final numerical answer as a number, nothing else.

Problem: {question}

Answer:"""


def cot_prompt(question):
    """Condition 2: Zero-shot Chain of Thought."""
    return f"""Solve the following math problem. Think step by step, then give the final numerical answer.
At the end of your response, write 'Final Answer: X' where X is the number.

Problem: {question}

Let's think step by step."""


def sc_prompt(question):
    """Condition 3: Self-Consistency (same as CoT, run 5 times, take majority)."""
    return f"""Solve the following math problem. Think step by step, then give the final numerical answer.
At the end of your response, write 'Final Answer: X' where X is the number.

Problem: {question}

Let's think step by step."""


def get_all_prompts(question):
    """Returns all three prompts for a given question."""
    return {
        "standard": standard_prompt(question),
        "cot": cot_prompt(question),
        "sc": sc_prompt(question)
    }