import json

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from tool.models import Solution


SOLUTION_REQUIREMENT_PROMPT = """
You are a helpful assistant, that analyzes task to be solved and thinks about requirements on the solution.

I will give you the following information:
Task: ...
Context: ...

You will responds with the pythonic list of requirements that the solution must meet:
["requirement1", "requirement2", ...]

Do not write anything else.
"""

_model = ChatOpenAI(model="gpt-4o-mini")


def get_requirements(empty_solution: Solution) -> Solution:
    task_str = f"Task: {empty_solution.task}\nContext: {empty_solution.context}"
    messages = [SystemMessage(content=SOLUTION_REQUIREMENT_PROMPT), HumanMessage(content=task_str)]
    result = _model.invoke(messages)
    return Solution(
        task=empty_solution.task,
        context=empty_solution.context,
        requirements=list(json.loads(str(result.content))),
    )
