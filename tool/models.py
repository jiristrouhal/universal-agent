from __future__ import annotations
from uuid import uuid4
from typing import Annotated, TypedDict, Literal, Any
from operator import add

import pydantic
from langchain_core.messages import AnyMessage


ResourceForm = Literal["code", "text"]
TestForm = Literal["code", "text"]
TestResult = Literal["pass", "fail", "unknown"]


EMPTY_RESOURCE = "Not provided."


class State(TypedDict):
    """This class represents an input and output state of the whole graph."""

    messages: Annotated[list[AnyMessage], add]


class Solution(pydantic.BaseModel):
    """This class represents a solution to a specific task in a specific context. It is used to store the solution in the database
    with the data necessary for the solution modifications and verification, including tests, source links and the solution structure.
    """

    context: str
    task: str
    requirements: list[str] = pydantic.Field(default_factory=list)
    solution_structure: list[str] = pydantic.Field(default_factory=list)
    resources: dict[str, str] = pydantic.Field(default_factory=dict)
    tests: list[Test] = pydantic.Field(default_factory=list)
    form: Literal["text", "code"] = "text"
    solution: str = ""
    similar_solutions: str = ""
    id: str = str(uuid4())
    proposal_tries: int = 0

    def empty(self) -> bool:
        return not bool(self.solution.strip())

    @property
    def task_description(self) -> str:
        return f"Task: {self.task}\nRequirements: {', '.join(self.requirements)}"


class Test(pydantic.BaseModel):
    """This class represents a test case for a solution."""

    description: str
    implementation: str = ""
    form: TestForm = "text"
    last_output: str = ""
    critique_of_last_run: str = ""
    result: TestResult = "unknown"


class SolutionWithTestsToRun(pydantic.BaseModel):
    task: str
    context: str
    requirements: list[str]
    resources: dict[str, str]
    solution_structure: list[str]
    tests_to_run: dict[int, Test]
    run_tests: dict[int, Test]
    form: ResourceForm = "text"
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
            resources=solution.resources,
            requirements=solution.requirements,
            solution_structure=solution.solution_structure,
            tests_to_run=tests_to_run,
            run_tests=run_tests,
            form=solution.form,
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
            resources=solution_with_next_test.resources,
            solution_structure=solution_with_next_test.solution_structure,
            tests=tests,
            form=solution_with_next_test.form,
            solution=solution_with_next_test.solution,
        )


class Resource(pydantic.BaseModel):
    form: ResourceForm
    context: str
    request: str
    content: str
    origin: str = "unknown"
    id: str = str(uuid4())
