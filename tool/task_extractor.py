from __future__ import annotations

import pydantic
from langchain_openai import ChatOpenAI
from trustcall import create_extractor
from langchain_core.messages import SystemMessage
from langgraph.graph import MessagesState


class Task(pydantic.BaseModel):
    task: str
    context: TaskContext


class TaskContext(pydantic.BaseModel):
    context: str
    system_state: str


model = ChatOpenAI(name="gpt-4o-mini")
task_extractor = create_extractor(model, tools=[Task], tool_choice="Task")
TASK_EXTRACTOR_PROMPT = "Extract a task with context from the following messages"


def parse_task(state: MessagesState) -> MessagesState:
    messages = [SystemMessage(TASK_EXTRACTOR_PROMPT)]
    messages.extend(state["messages"])  # type: ignore
    result = task_extractor.invoke({"messages": messages})
    return MessagesState({"messages": result["messages"]})
