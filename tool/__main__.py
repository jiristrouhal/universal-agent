from __future__ import annotations
from typing import Any

import pydantic
from IPython.display import Image
import dotenv
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from .task_extractor import parse_task


dotenv.load_dotenv()


builder = StateGraph(MessagesState)
builder.add_node("parse_task", parse_task)
builder.add_edge(START, "parse_task")
builder.add_edge("parse_task", END)
graph = builder.compile()


# View the graph
with open("misc/graph.png", "wb") as f:
    f.write(Image(graph.get_graph().draw_mermaid_png()).data)


result = graph.invoke(
    {"messages": [HumanMessage(content="I would like to build a house, can you help me?")]}
)
print(result)
