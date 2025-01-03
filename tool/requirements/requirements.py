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

Also include requirement, that all facts in the solution are based on collected Resources.

Do not write anything else.
"""


_model = ChatOpenAI(model="gpt-4o-mini")


def get_requirements(empty_solution: Solution) -> Solution:
    task_str = f"Task: {empty_solution.task}\nContext: {empty_solution.context}"
    messages = [SystemMessage(content=SOLUTION_REQUIREMENT_PROMPT), HumanMessage(content=task_str)]
    result = _model.invoke(messages)
    reqs = list(json.loads(str(result.content)))
    empty_solution.requirements = reqs
    return empty_solution
