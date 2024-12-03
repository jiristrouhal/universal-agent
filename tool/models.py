from __future__ import annotations
from typing import Annotated, TypedDict
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
    tests: str


class TaskWithSolutionDraft(pydantic.BaseModel):
    task: str
    context: str
    requirements: list[str]
    tests: str
    solution_draft: str


class TaskWithSourceAugmentedDraft(pydantic.BaseModel):
    task: str
    context: str
    requirements: list[str]
    tests: str
    solution_draft: str
    sources: dict[str, str]


def task_with_empty_recall(task: TaskPlain) -> TaskWithSolutionRecall:
    return TaskWithSolutionRecall(task=task.task, context=task.context, solution_recall="")
