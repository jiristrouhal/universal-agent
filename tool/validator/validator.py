# look at the solution and tests

# for each test:
# 1. assess if the test can be currently run
# 2. choose the test implementation accordingly to the solution format
# 3. write the test implementation
# 4. run the test
# 5. provide a critique for the result

# store the solution with updated test critiques
from __future__ import annotations


import dotenv
from pprint import pprint
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun, WikipediaAPIWrapper
from langchain_experimental.tools import PythonREPLTool
from langchain_core.tools import tool

from tool.solution_proposer.solution import Solution, Test


dotenv.load_dotenv()
_model = ChatOpenAI(model="gpt-4o-mini")
_test_runner = create_react_agent(
    _model, tools=[WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()), PythonREPLTool()]
)

TEST_FORM_PROMPT = """
You are a helpful assistant, that helps to write a test for a given solution.

You will be provided with the following information:

Test description: ...
Solution: ...

Your task is to determine the form of the test implementation. Your response should be only a single word:
- 'text' if the test is a text-based
- 'code' if the test is a code-based

DO NOT write anything else.
"""


TEST_IMPLEMENTER_PROMPT = """
You are a helpful assistant, that implements a single test according to the solution format and the provided test description.

You will be provided with the following information:
Test form: ...
Test description: ...
Solution: ...

Your task is to implement the test according to the description and run it on the solution based on the solution format.

Write only a single test copmlying with the Test description. Do not write multiple tests. Do not write anything else.

"""

TEST_RUNNER_PROMPT = """
You are a helpful assistant, that runs a test on a given solution.

You will be provided with the following information:
Test implementation: ...
Solution: ...

Run only the test implementation. Do not run anythinig else. Do not create any new test cases.

Your task is to run the test implementation on the solution and provide a critique of the result. The critique will contain
- if the single test passed or failed
In case the test failed, the critique will contain also
- the reason of the failure
- the part of solution that caused the failure
- suggestions for making the solution pass the test
"""


def implement_test(test: Test, solution: str) -> None:
    """Implements a test on a solution."""
    test_form = _determine_test_form(test, solution)
    query = f"Test form: {test_form}, Test description: {test.description}\nSolution: {solution}"
    messages = [SystemMessage(content=TEST_IMPLEMENTER_PROMPT), HumanMessage(content=query)]
    response = str(_model.invoke(messages).content)
    test.implementation = response


def _determine_test_form(test: Test, solution: str) -> str:
    """Determines the form of the test implementation."""
    query = f"Test description: {test.description}\nSolution: {solution}"
    messages = [SystemMessage(content=TEST_FORM_PROMPT), HumanMessage(content=query)]
    response = str(_model.invoke(messages).content)
    return response


def run_test(test: Test, solution: str) -> None:
    """Runs a single test on a solution."""
    query = f"Test implementation: {test.implementation}\nSolution: {solution}"
    messages = [SystemMessage(content=TEST_RUNNER_PROMPT), HumanMessage(content=query)]
    response = str(_test_runner.invoke({"messages": messages})["messages"][-1].content)
    test.critique_of_last_run = response
