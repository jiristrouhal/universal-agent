from __future__ import annotations
from typing import Literal

import dotenv
import pydantic
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from tool.models import Solution as _Solution, Test as _Test


dotenv.load_dotenv()


class SolutionWithTestsToRun(pydantic.BaseModel):
    task: str
    context: str
    requirements: list[str]
    sources: dict[str, str]
    solution_structure: list[str]
    tests_to_run: dict[int, _Test]
    run_tests: dict[int, _Test]
    solution: str

    @staticmethod
    def fromSolution(solution: _Solution) -> SolutionWithTestsToRun:
        tests_to_run, run_tests = dict(), dict()
        for t in solution.tests:
            if t.critique_of_last_run:
                run_tests[solution.tests.index(t)] = t
            else:
                tests_to_run[solution.tests.index(t)] = t
        return SolutionWithTestsToRun(
            task=solution.task,
            context=solution.context,
            sources=solution.resources,
            requirements=solution.requirements,
            solution_structure=solution.solution_structure,
            tests_to_run=tests_to_run,
            run_tests=run_tests,
            solution=solution.solution,
        )

    @staticmethod
    def toSolution(solution_with_next_test: SolutionWithTestsToRun) -> _Solution:
        all_tests = solution_with_next_test.tests_to_run.copy()
        all_tests.update(solution_with_next_test.run_tests)
        tests = [all_tests[k] for k in sorted(all_tests.keys())]
        return _Solution(
            task=solution_with_next_test.task,
            requirements=solution_with_next_test.requirements,
            context=solution_with_next_test.context,
            resources=solution_with_next_test.sources,
            solution_structure=solution_with_next_test.solution_structure,
            tests=tests,
            solution=solution_with_next_test.solution,
        )


QUESTION_FORMULATION_PROMPT = """
You are a helpful assistant, that helps to convert a test description into specific questions, that will be answered by someone else.

The test is:
{test_description}

Write questions, that will be answered by someone else. Answers to these questions will verify the solution.

Example 1:
The test is:
 "The solution must contain a sentence about the Czech Republic.",
Question:
 "The solution must contain a sentence about the Czech Republic. Does the solution contain a sentence about the Czech Republic?"

Example 2:
The test is:
 "Alice not being in the list is a mistake.",
Question:
 "Alice not being in the list is a mistake. Is Alice in the list?"

Write only the questions. Do not write anything else.
"""


TEST_RUNNER_PROMPT = """
You are a helpful assistant, that helps to verify the correctness of the solution by running the test.

Here is a solution to some task:
{solution}

You need to provide answers to these questions related to the solution:
{questions}

Write only the answers. Do not write anything else.
"""


CRITIC_PROMPT = """
You are a critic, that helps me to improve the solution. I will always provide you with a following information:

Solution: ...
Test description: ...
Test implementation: ...
Test result: ...

Please, provide me with a critique of the solution. The critique should contain
- any error that occured, if applicable,
- place in the solution, where the error is,
- a single-sentence suggestion, how to fix the error.

If all the answers received positive responses, write also "TEST_PASSED".

Do not write anything else.
"""


TEXT_TESTER_END = "__text_tester_end__"


_model = ChatOpenAI(name="gpt-4o-mini")


def prepareSolution_with_tests_to_run(solution: _Solution) -> SolutionWithTestsToRun:
    return SolutionWithTestsToRun.fromSolution(solution)


def pick_test(solution_with_next_test: SolutionWithTestsToRun) -> SolutionWithTestsToRun:
    return solution_with_next_test


def any_next_test(
    solution_with_next_test: SolutionWithTestsToRun,
) -> Literal["end", "next_test"]:
    if solution_with_next_test.tests_to_run:
        return "next_test"
    return "end"


def returnSolution_with_updated_tests(
    solution_with_next_test: SolutionWithTestsToRun,
) -> _Solution:
    return SolutionWithTestsToRun.toSolution(solution_with_next_test)


def implement_test(solution_with_next_test: SolutionWithTestsToRun) -> SolutionWithTestsToRun:
    """Provide the implementation for the selected test identified by its id in the list of all tests.

    The implementation is list of questions that will be answered to verify the correctness of the solution.
    """
    assert isinstance(solution_with_next_test, SolutionWithTestsToRun)
    test_id = list(solution_with_next_test.tests_to_run.keys())[0]
    assert test_id is not None
    test = solution_with_next_test.tests_to_run[test_id]
    prompt = QUESTION_FORMULATION_PROMPT.format(test_description=test.description)
    response = _model.invoke([SystemMessage(content=prompt)])
    solution_with_next_test.tests_to_run[test_id].implementation = str(response.content)
    return solution_with_next_test


def run_test(solution_with_next_test: SolutionWithTestsToRun) -> SolutionWithTestsToRun:
    """Run the test identified by its id in the list of all tests.

    The test is run by asking the questions that were formulated during the implementation phase.
    """
    test_id = list(solution_with_next_test.tests_to_run.keys())[0]
    test = solution_with_next_test.tests_to_run.pop(test_id)
    prompt = TEST_RUNNER_PROMPT.format(
        solution=solution_with_next_test.solution, questions=test.implementation
    )
    response = _model.invoke([SystemMessage(content=prompt)])
    test.last_output = str(response.content)
    solution_with_next_test.run_tests[test_id] = test
    return solution_with_next_test


def criticize(solution: _Solution) -> _Solution:
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


def get_text_validator_builder() -> StateGraph:
    text_validator_builder = StateGraph(_Solution)
    text_validator_builder.add_node(
        "prepareSolution_with_no_test_to_run_next", prepareSolution_with_tests_to_run
    )
    text_validator_builder.add_node("pick_test", pick_test)
    text_validator_builder.add_node("implement_next_test", implement_test)
    text_validator_builder.add_node("run_test", run_test)
    text_validator_builder.add_node(
        "returnSolution_with_updated_tests", returnSolution_with_updated_tests
    )
    text_validator_builder.add_node("critic", criticize)

    text_validator_builder.add_edge(START, "prepareSolution_with_no_test_to_run_next")
    text_validator_builder.add_edge("prepareSolution_with_no_test_to_run_next", "pick_test")
    text_validator_builder.add_conditional_edges(
        "pick_test",
        any_next_test,
        path_map={
            "end": "returnSolution_with_updated_tests",
            "next_test": "implement_next_test",
        },
    )
    text_validator_builder.add_edge("implement_next_test", "run_test")
    text_validator_builder.add_edge("run_test", "pick_test")
    text_validator_builder.add_edge("returnSolution_with_updated_tests", "critic")
    text_validator_builder.add_edge("critic", END)
    return text_validator_builder
