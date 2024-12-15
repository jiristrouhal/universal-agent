from __future__ import annotations

import dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt.chat_agent_executor import create_react_agent, CompiledGraph
from langchain_core.tools import BaseTool
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun, WikipediaAPIWrapper
from langchain_experimental.tools import PythonREPLTool

from tool.models import (
    Solution as _Solution,
    Test as _Test,
    TestResult as _TestResult,
    TestForm as _TestForm,
)


dotenv.load_dotenv()


TEST_FORM_PROMPT = """
You are a helpful assistant, that helps to write a test for a given solution.

You will be provided with the following information:

Test description: ...
Tested solution: ...

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
Tested solution: ...

Your task is to implement the test according to the Test description and the required Test form.
Include only values and assertions from the Test description. Do not make any additional assumptions.

Write only a single test complying with the Test description.
Do not write multiple tests. Do not write anything else.
"""

TEST_RUNNER_PROMPT = """
You are a helpful assistant, that runs a test on a given solution.

You will be provided with the following information:
Test form: ...
Test implementation: ...
Tested solution: ...

Run only the test implementation. Do not run anything else.
Do not create any new test cases.

Your task is to run the test implementation on the solution and provide a critique of the result. The critique will contain
- if the single test passed or failed. If it passsed include 'PASSED' in the critique, if it failed include 'FAILED' in the critique
In case the test failed, the critique will contain also
- the reason of the failure
- the part of solution that caused the failure
- suggestions for making the solution pass the test
"""


class Validator:

    def __init__(self, openai_model: str = "gpt-4o-mini") -> None:
        self._model = ChatOpenAI(model=openai_model)
        common_tools = [WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())]
        text_tools: list[BaseTool] = []
        code_tools = [PythonREPLTool()]
        self._runner: dict[_TestForm, CompiledGraph] = {
            "text": create_react_agent(self._model, tools=common_tools + text_tools),
            "code": create_react_agent(self._model, tools=common_tools + code_tools),
        }

    def review(self, solution: _Solution) -> None:
        """Reviews a solution."""
        for test in solution.tests:
            self.implement_test(test, solution.solution)
            self.run_test(test, solution.solution)

    def implement_test(self, test: _Test, solution: str) -> None:
        """Implements a test on a solution."""
        test.form = self._determine_test_form(test, solution)
        query = f"Test form: {test.form}\nTest description: {test.description}\nTested solution: {solution}"
        messages = [SystemMessage(content=TEST_IMPLEMENTER_PROMPT), HumanMessage(content=query)]
        response = str(self._model.invoke(messages).content)
        test.implementation = response

    def _determine_test_form(self, test: _Test, solution: str) -> _TestForm:
        """Determines the form of the test implementation."""
        query = f"Test description: {test.description}\nTested solution: {solution}"
        messages = [SystemMessage(content=TEST_FORM_PROMPT), HumanMessage(content=query)]
        response = str(self._model.invoke(messages).content)
        return "code" if response == "code" else "text"

    def run_test(self, test: _Test, solution: str) -> None:
        """Runs a single test on a solution."""
        query = f"Test form: {test.form}\nTest implementation: {test.implementation}\nTested solution: {solution}"
        messages = [SystemMessage(content=TEST_RUNNER_PROMPT), HumanMessage(content=query)]
        response = str(
            self._runner[test.form].invoke({"messages": messages})["messages"][-1].content
        )
        test.critique_of_last_run = response
        test.result = self._determine_test_run_result(response)

    def _determine_test_run_result(self, critique: str) -> _TestResult:
        """Determines if the test passed or failed."""
        if "PASSED" in critique:
            return "pass"
        elif "FAILED" in critique:
            return "fail"
        return "unknown"
