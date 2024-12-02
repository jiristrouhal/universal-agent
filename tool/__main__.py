from __future__ import annotations

from IPython.display import Image
import dotenv
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage

from tool.models import State
from tool.task_extractor import parse_task
from tool.recaller import recall


dotenv.load_dotenv()


builder = StateGraph(State)
builder.add_node("parse_task", parse_task)
builder.add_node("recall_solutions", recall)

builder.add_edge(START, "parse_task")
builder.add_edge("parse_task", "recall_solutions")
builder.add_edge("recall_solutions", END)
graph = builder.compile()


# View the graph
with open("misc/graph.png", "wb") as f:
    f.write(Image(graph.get_graph().draw_mermaid_png()).data)


result = graph.invoke(
    {"messages": [HumanMessage(content="I would like to build a house, can you help me?")]}
)
for m in result["messages"]:
    print(m.pretty_print())
