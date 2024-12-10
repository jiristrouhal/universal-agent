from __future__ import annotations
from typing import Literal, Optional

import dotenv
import pydantic
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from tool.solver.solution import Solution, Test


dotenv.load_dotenv()


class SolutionWithTestsToRun(pydantic.BaseModel):
    task: str
    context: str
    requirements: list[str]
    sources: dict[str, str]
    solution_structure: list[str]
    tests_to_run: dict[int, Test]
    run_tests: dict[int, Test]
    solution: str

    @staticmethod
    def from_solution(solution: Solution) -> SolutionWithTestsToRun:
        tests_to_run, run_tests = dict(), dict()
        for t in solution.tests:
            if t.critique_of_last_run:
                run_tests[solution.tests.index(t)] = t
            else:
                tests_to_run[solution.tests.index(t)] = t
        return SolutionWithTestsToRun(
            task=solution.task,
            context=solution.context,
            sources=solution.sources,
            requirements=solution.requirements,
            solution_structure=solution.solution_structure,
            tests_to_run=tests_to_run,
            run_tests=run_tests,
            solution=solution.solution,
        )

    @staticmethod
    def to_solution(solution_with_next_test: SolutionWithTestsToRun) -> Solution:
        all_tests = solution_with_next_test.tests_to_run.copy()
        all_tests.update(solution_with_next_test.run_tests)
        tests = [all_tests[k] for k in sorted(all_tests.keys())]
        return Solution(
            task=solution_with_next_test.task,
            requirements=solution_with_next_test.requirements,
            context=solution_with_next_test.context,
            sources=solution_with_next_test.sources,
            solution_structure=solution_with_next_test.solution_structure,
            tests=tests,
            solution=solution_with_next_test.solution,
        )


QUESTION_FORMULATION_PROMPT = """
You are a helpful assistant, that helps to convert a test description into specific questions, that will be answered by someone else.

The test is:
{test_description}

Write questions, that will be answered by someone else. Answers to these questions will verify the solution.

For example, if the test is: "The solution must contain a sentence about the Czech Republic.", the question can be: "Does the solution contain a sentence about the Czech Republic?"

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


TEXT_TESTER_END = "__text_tester_end__"


_model = ChatOpenAI(name="gpt-4o-mini")


def prepare_solution_with_tests_to_run(solution: Solution) -> SolutionWithTestsToRun:
    return SolutionWithTestsToRun.from_solution(solution)


def pick_test(solution_with_next_test: SolutionWithTestsToRun) -> SolutionWithTestsToRun:
    print(solution_with_next_test.tests_to_run)
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
    test.last_result = str(response.content)
    solution_with_next_test.run_tests[test_id] = test
    return solution_with_next_test


text_validator_builder = StateGraph(Solution)

text_validator_builder.add_node(
    "prepare_solution_with_no_test_to_run_next", prepare_solution_with_tests_to_run
)
text_validator_builder.add_node("pick_test", pick_test)
text_validator_builder.add_node("implement_next_test", implement_test)
text_validator_builder.add_node("run_test", run_test)
text_validator_builder.add_node(
    "return_solution_with_updated_tests", return_solution_with_updated_tests
)

text_validator_builder.add_edge(START, "prepare_solution_with_no_test_to_run_next")
text_validator_builder.add_edge("prepare_solution_with_no_test_to_run_next", "pick_test")
text_validator_builder.add_conditional_edges(
    "pick_test",
    any_next_test,
    path_map={
        "end": "return_solution_with_updated_tests",
        "next_test": "implement_next_test",
    },
)
text_validator_builder.add_edge("implement_next_test", "run_test")
text_validator_builder.add_edge("run_test", "pick_test")
text_validator_builder.add_edge("return_solution_with_updated_tests", END)
