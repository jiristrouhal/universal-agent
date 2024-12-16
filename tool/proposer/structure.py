import json

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from tool.models import Solution, Solution


SOLUTION_PROMPT = """
You are a helpful assistant, that helps me to design solution to a task.

I will give you the following information:

Task: ...
Context: ...
Requirements: ...
Tests: ...

Write down step by step the solution structure. This includes the solution to the problem itself, not including tests, sources or reasoning.

Please, respond only with the parts of the solution structure each formatted as a plain text, one after another, forming a list:
["Part 1", "Part 2", "Part 3", ...]

Do not write anything else.
"""


_model = ChatOpenAI(model="gpt-4o-mini")


def draft_solution(solution: Solution) -> Solution:
    task_str = (
        f"Task: {solution.task}\n"
        f"Context: {solution.context}\n"
        f"Requirements: {solution.requirements}\n"
        f"Tests: {solution.tests}"
    )
    messages = [SystemMessage(content=SOLUTION_PROMPT), HumanMessage(content=task_str)]
    result = _model.invoke(messages)
    solution.solution_structure = list(json.loads(str(result.content)))
    return solution
