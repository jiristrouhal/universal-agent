import json

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from tool.logs import get_logger
from tool.models import Solution, Test


logger = get_logger()


SOLUTION_REQUIREMENT_PROMPT = """
You are a helpful assistant, that helps me to verify solution to a given task. I will give you the task, a context and list of requirements on the solution.

I will give you the following information:

Task: ...
Context: ...
Requirements: ...

Please, follow these guidelines when writing the tests:
0) If the task contains any tests, include them in your response. Assume, that these tests are literally correct.
1) Each test correspond to a single requirement.
2) There can be multiple distinct tests for a single requirement.
3) Each test must be independent from the others.
4) Each test must be deterministic.
5) Do not include additional constraints or assumptions in the tests other than specified in the requirements or tests in the task.

Please, respond with the descriptions of the tests (what is tested, inputs and expected results if applicable) formatted as list with items formatter as a plain text, one after another. The list must contain at least one test:
["Test 1", "Test 2", "Test 3", ...]

Do not write anything else.
"""


_model = ChatOpenAI(model="gpt-4o-mini")


def get_tests(solution: Solution) -> Solution:
    task_str = (
        f"Task: {solution.task}\nContext: {solution.context}\nRequirements: {solution.requirements}"
    )
    messages = [SystemMessage(content=SOLUTION_REQUIREMENT_PROMPT), HumanMessage(content=task_str)]
    result = _model.invoke(messages)
    tests_str = list(json.loads(str(result.content)))
    tests = [Test(description=t, form=solution.form) for t in tests_str]
    solution.tests.extend(tests)
    logger.info(
        f"Updated tests for solution of task '{solution.task}': {[t.description for t in solution.tests]}"
    )
    return solution
