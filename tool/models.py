from __future__ import annotations
from typing import Annotated, TypedDict
from operator import add

import pydantic
from langchain_core.messages import AnyMessage


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add]


class Task(pydantic.BaseModel):
    task: str
    context: str
