from typing import Literal

import dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_experimental.utilities import PythonREPL

from tool.models import Solution, SolutionWithTestsToRun


dotenv.load_dotenv()


TEST_CODE_WRITER_PROMPT = """
You are a helpful assistant, that helps to convert a test description into a test code, that verifies the correctness of the solution.

The test description is:
{test_description}
The tested solution is:
{solution}

Repond with only a python code wihout any code block marks. The reponse will in plain text contain (in the same order):
- the pasted solution code,
- the test code,
- call of the test code.
The result must be self-contained and executable in python REPL.

Add necessary comments explaining the test code. At the end, there must be a print statement printing the test result.
Use ordinary assert statements, avoid using pytest or unittest. Add string after asserted statement to indicate the test description.
When asseting equality of two floats, allow for a small error margin.

Do not write anything else.
"""


CRITIC_PROMPT = """
You are a critic, that helps me to improve the solution. I will give you the following information:

Solution: ...
Test description: ...
Test implementation: ...
Test result: ...

Please, provide me with a critique of the solution. The critique should contain
- any error that occured, if applicable,
- place in the solution, where the error is,
- a single-sentence suggestion, how to fix the error.

If the test matches expectations from test description, write also "TEST_PASSED".

Do not write anything else.
"""


TEXT_TESTER_END = "__text_tester_end__"


_model = ChatOpenAI(name="gpt-4o-mini")
_python_repl = PythonREPL()


def prepare_solution_with_tests_to_run(solution: Solution) -> SolutionWithTestsToRun:
    return SolutionWithTestsToRun.from_solution(solution)


def pick_test(solution_with_next_test: SolutionWithTestsToRun) -> SolutionWithTestsToRun:
    return solution_with_next_test


def any_next_test(
    solution_with_next_test: SolutionWithTestsToRun,
) -> Literal["end", "next_test"]:
    if solution_with_next_test.tests_to_run:
        return "next_test"
    return "end"


def return_solution_with_updated_tests(
    solution_with_next_test: SolutionWithTestsToRun,
) -> Solution:
    return SolutionWithTestsToRun.to_solution(solution_with_next_test)


def implement_test(solution_with_next_test: SolutionWithTestsToRun) -> SolutionWithTestsToRun:
    """Provide the implementation for the selected test identified by its id in the list of all tests.

    The implementation is list of questions that will be answered to verify the correctness of the solution.
    """
    assert isinstance(solution_with_next_test, SolutionWithTestsToRun)
    test_id = list(solution_with_next_test.tests_to_run.keys())[0]
    assert test_id is not None
    test = solution_with_next_test.tests_to_run[test_id]
    prompt = TEST_CODE_WRITER_PROMPT.format(
        test_description=test.description, solution=solution_with_next_test.solution
    )
    response = _model.invoke([SystemMessage(content=prompt)])
    solution_with_next_test.tests_to_run[test_id].implementation = str(response.content)
    return solution_with_next_test


def run_test(solution_with_next_test: SolutionWithTestsToRun) -> SolutionWithTestsToRun:
    """Run the test identified by its id in the list of all tests.

    The test is run by asking the questions that were formulated during the implementation phase.
    """
    test_id = list(solution_with_next_test.tests_to_run.keys())[0]
    test = solution_with_next_test.tests_to_run.pop(test_id)
    response = run_python_code(code=test.implementation)
    solution_with_next_test.run_tests[test_id] = test
    solution_with_next_test.run_tests[test_id].last_output = response
    return solution_with_next_test


def run_python_code(code: str) -> str:
    response = _python_repl.run(code, timeout=15)
    return str(response)


def criticize(solution: Solution) -> Solution:
    for test in solution.tests:
        query = (
            f"Solution: {solution.solution}\nTest description: {test.description}\n"
            f"Test implementation: {test.implementation}\nTest result: {test.last_output}"
        )
        response = _model.invoke(
            [SystemMessage(content=CRITIC_PROMPT), HumanMessage(content=query)]
        )
        test.critique_of_last_run = str(response.content)
        if "TEST_PASSED" in response.content:
            test.result = "pass"
        else:
            test.result = "fail"
    return solution


def get_code_validator_builder() -> StateGraph:
    builder = StateGraph(Solution)
    builder.add_node(
        "prepare_solution_with_no_test_to_run_next", prepare_solution_with_tests_to_run
    )
    builder.add_node("pick_test", pick_test)
    builder.add_node("implement_next_test", implement_test)
    builder.add_node("run_test", run_test)
    builder.add_node("return_solution_with_updated_tests", return_solution_with_updated_tests)
    builder.add_node("critic", criticize)

    builder.add_edge(START, "prepare_solution_with_no_test_to_run_next")
    builder.add_edge("prepare_solution_with_no_test_to_run_next", "pick_test")
    builder.add_conditional_edges(
        "pick_test",
        any_next_test,
        path_map={
            "end": "return_solution_with_updated_tests",
            "next_test": "implement_next_test",
        },
    )
    builder.add_edge("implement_next_test", "run_test")
    builder.add_edge("run_test", "pick_test")
    builder.add_edge("return_solution_with_updated_tests", "critic")
    builder.add_edge("critic", END)
    return builder
