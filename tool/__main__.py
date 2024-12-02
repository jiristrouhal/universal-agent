from __future__ import annotations

from IPython.display import Image
import dotenv
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage

from tool.models import State, task_with_empty_recall
from tool.task_extractor import parse_task
from tool.recaller import recall, recalled_or_new
from tool.requirements import get_requirements, print_requirements


dotenv.load_dotenv()


builder = StateGraph(State)
builder.add_node("parse_task", parse_task)
builder.add_node("init_solution_recall", task_with_empty_recall)
builder.add_node("recall_solutions", recall)
builder.add_node("get_requirements", get_requirements)
builder.add_node("print_requirements", print_requirements)

builder.add_edge(START, "parse_task")
builder.add_edge("parse_task", "init_solution_recall")
builder.add_edge("init_solution_recall", "recall_solutions")
builder.add_conditional_edges(
    "recall_solutions", recalled_or_new, {"new": "get_requirements", "recalled": END}
)
builder.add_edge("get_requirements", "print_requirements")
builder.add_edge("print_requirements", END)
graph = builder.compile()


# View the graph
with open("misc/graph.png", "wb") as f:
    f.write(Image(graph.get_graph().draw_mermaid_png()).data)


result = graph.invoke(
    {"messages": [HumanMessage(content="I would like to build a house, can you help me?")]}
)

for m in result["messages"]:
    print(m.pretty_print())
