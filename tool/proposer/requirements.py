import json

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from tool.models import Solution


SOLUTION_REQUIREMENT_PROMPT = """
You are a helpful assistant, that analyzes task to be solved and thinks about requirements on the solution.

I will give you the following information:
Task: ...
Context: ...

You will responds with a list of requirements that the solution must meet.
["requirement1", "requirement2", ...]

Each requirement describes some aspect of the task. Each requirement should be unique and specific. If the response includes some statements with
meaning similar to for example:
- 'this must comply with ...',
- 'this must pass ...',
- 'this must be ...',
- 'this must have ...',
- 'this must not ...',
- 'assert that ...',
etc., include them in the requirements.

Do not write anything else.
"""

REQUIREMENT_FIX_PROMPT = """
You are a helpful assistant, that helps me to fix the requirements on the solution, if they do not correspond to the task.

I will give you the following information:
Task: ...
Context: ...
Requirement: ...

You will respond with the same requirement, if it is correct, or with the corrected requirement, if it is incorrect.
Respond to me only with the requirement content. Do not write anything else.
"""


_model = ChatOpenAI(model="gpt-4o-mini")


def get_requirements(empty_solution: Solution) -> Solution:
    task_str = f"Task: {empty_solution.task}\nContext: {empty_solution.context}"
    messages = [SystemMessage(content=SOLUTION_REQUIREMENT_PROMPT), HumanMessage(content=task_str)]
    result = _model.invoke(messages)
    reqs = list(json.loads(str(result.content)))
    for k in range(len(reqs)):
        query = f"Task: {empty_solution.task}\nContext: {empty_solution.context}\nRequirement: {reqs[k]}"
        messages = [SystemMessage(content=REQUIREMENT_FIX_PROMPT), HumanMessage(content=query)]
        result = _model.invoke(messages)
        reqs[k] = str(result.content)
    return Solution(
        task=empty_solution.task,
        context=empty_solution.context,
        requirements=reqs,
    )
