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
    requirements: str


class TaskWithTests(pydantic.BaseModel):
    task: str
    context: str
    requirements: str
    tests: str


def task_with_empty_recall(task: TaskPlain) -> TaskWithSolutionRecall:
    return TaskWithSolutionRecall(task=task.task, context=task.context, solution_recall="")
