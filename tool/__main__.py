from __future__ import annotations

from IPython.display import Image
import dotenv
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, BaseMessage

from tool.models import State, task_with_empty_recall
from tool.task_extractor import parse_task
from tool.recaller import recall, recalled_or_new
from tool.solver.requirements import get_requirements
from tool.solver.tests import get_tests
from tool.solver.structure import get_solution_structure
from tool.solver.sources import collect_sources
from tool.solver.solution import propose_solution, print_solution


dotenv.load_dotenv()


builder = StateGraph(State)


builder.add_node("parse_task", parse_task)
builder.add_node("init_solution_recall", task_with_empty_recall)
builder.add_node("recall_solutions", recall)
builder.add_node("get_requirements", get_requirements)
builder.add_node("get_tests", get_tests)
builder.add_node("get_solution_structure", get_solution_structure)
builder.add_node("collect_resources", collect_sources)
builder.add_node("propose_solution", propose_solution)
builder.add_node("print_solution", print_solution)


builder.add_edge(START, "parse_task")
builder.add_edge("parse_task", "init_solution_recall")
builder.add_edge("init_solution_recall", "recall_solutions")
builder.add_conditional_edges(
    "recall_solutions", recalled_or_new, {"new": "get_requirements", "recalled": END}
)
builder.add_edge("get_requirements", "get_tests")
builder.add_edge("get_tests", "get_solution_structure")
builder.add_edge("get_solution_structure", "collect_resources")
builder.add_edge("collect_resources", "propose_solution")
builder.add_edge("propose_solution", "print_solution")
builder.add_edge("print_solution", END)
graph = builder.compile()


# View the graph
with open("misc/graph.png", "wb") as f:
    f.write(Image(graph.get_graph().draw_mermaid_png()).data)


result = graph.invoke(
    {
        "messages": [
            HumanMessage(
                content="""
                    Can you write for me a function that determines temperature distribution in a rod constantly heated on one side and cooled on the other?
                    There are no heat losses or sources along the rod. Consider the rod one-dimensional. I assume some constant initial temperature along the rod.
                """
            )
        ]
    }
)

for m in result["messages"]:
    assert isinstance(m, BaseMessage)
    m.pretty_print()
