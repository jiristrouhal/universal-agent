from __future__ import annotations
from uuid import uuid4
from typing import Annotated, TypedDict, Literal
from operator import add

import pydantic
from langchain_core.messages import AnyMessage


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add]


class TaskPlain(pydantic.BaseModel):
    task: str
    context: str


class TaskWithSolutionRecall(pydantic.BaseModel):
    task: str
    context: str
    solution_recall: str


class TaskToSolve(pydantic.BaseModel):
    task: str
    context: str
    requirements: list[str]


class TaskWithTests(pydantic.BaseModel):
    task: str
    context: str
    requirements: list[str]
    tests: list[str]


class TaskWithSolutionStructure(pydantic.BaseModel):
    task: str
    context: str
    requirements: list[str]
    tests: list[str]
    solution_structure: list[str]


class TaskWithSources(pydantic.BaseModel):
    task: str
    context: str
    requirements: list[str]
    tests: list[str]
    solution_structure: list[str]
    sources: dict[str, str]


class Solution(pydantic.BaseModel):
    """This class represents a solution to a specific task in a specific context. It is used to store the solution in the database
    with the data necessary for the solution modifications and verification, including tests, source links and the solution structure.
    """

    context: str
    task: str
    requirements: list[str]
    solution_structure: list[str]
    sources: dict[str, str]
    tests: list[Test]
    form: Literal["text", "code"] = "text"
    solution: str
    id: str = str(uuid4())

    @property
    def task_description(self) -> str:
        return f"Context: {self.context}\nTask: {self.task}"


class Test(pydantic.BaseModel):
    """This class represents a test case for a solution."""

    description: str
    implementation: str = ""
    last_output: str = ""
    critique_of_last_run: str = ""
    result: Literal["pass", "fail", "unknown"] = "unknown"


class SolutionWithTestsToRun(pydantic.BaseModel):
    task: str
    context: str
    requirements: list[str]
    sources: dict[str, str]
    solution_structure: list[str]
    tests_to_run: dict[int, Test]
    run_tests: dict[int, Test]
    form: Literal["text", "code"] = "text"
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
            sources=solution_with_next_test.sources,
            solution_structure=solution_with_next_test.solution_structure,
            tests=tests,
            form=solution_with_next_test.form,
            solution=solution_with_next_test.solution,
        )


ResourceForm = Literal["code", "text"]


class Resource(pydantic.BaseModel):
    form: ResourceForm
    context: str
    query: str
    content: str
    id: str = str(uuid4())


def task_with_empty_recall(task: TaskPlain) -> TaskWithSolutionRecall:
    return TaskWithSolutionRecall(task=task.task, context=task.context, solution_recall="")
