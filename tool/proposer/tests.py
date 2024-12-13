import json

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from tool.models import Solution, Test


SOLUTION_REQUIREMENT_PROMPT = """
You are a helpful assistant, that helps me to verify solution to a given task. I will give you the task, a context and list of requirements on the solution.

I will give you the following information:

Task: ...
Context: ...
Requirements: ...

Please, follow these guidelines when writing the tests:
    1) Each test correspond to a single requirement.
    2) There can be multiple distinct tests for a single requirement.
    3) Each test must be independent from the others.
    4) Each test must be deterministic.

Please, respond with the descriptions of the tests (what is tested, inputs and expected results if applicable) formatted as list with items formatter as a plain text, one after another. The list must contain at least one test:
["Test 1", "Test 2", "Test 3", ...]


Do not write anything else.
"""

_model = ChatOpenAI(model="gpt-4o-mini")


def get_tests(task_to_solve: Solution) -> Solution:
    task_str = f"Task: {task_to_solve.task}\nContext: {task_to_solve.context}\nRequirements: {task_to_solve.requirements}"
    messages = [SystemMessage(content=SOLUTION_REQUIREMENT_PROMPT), HumanMessage(content=task_str)]
    result = _model.invoke(messages)
    tests_str = list(json.loads(str(result.content)))
    tests = [Test(description=t) for t in tests_str]
    return Solution(
        task=task_to_solve.task,
        context=task_to_solve.context,
        requirements=task_to_solve.requirements,
        tests=tests,
    )
