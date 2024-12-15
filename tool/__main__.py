from __future__ import annotations

from IPython.display import Image
import dotenv
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, BaseMessage

from tool.parsers import task_parser
from tool.proposer.requirements import get_requirements
from tool.proposer.tests import get_tests
from tool.proposer.structure import draft_solution
from tool.proposer import Proposer
from tool.proposer.resources import ResourceManager
from tool.memory.recaller import Recaller
from tool.models import State, Solution, Solution

dotenv.load_dotenv()


builder = StateGraph(State)
recaller = Recaller(db_dir_path="./data/solutions")
info_manager = ResourceManager(db_dir_path="./data/resources")
proposer = Proposer(db_dir_path="./data/solutions")


builder.add_node("parse_task", task_parser)
builder.add_node("get_requirements", get_requirements)
builder.add_node("recall", recaller.recall, input=Solution)
builder.add_node("get_tests", get_tests)
builder.add_node("draft_solution", draft_solution)
builder.add_node("get_resources", info_manager.get_resources, input=Solution)
builder.add_node("propose_solution", proposer.propose_solution, input=Solution)
builder.add_node("print_solution", proposer.print_solution, input=Solution)


builder.add_edge(START, "parse_task")
builder.add_edge("parse_task", "get_requirements")
builder.add_edge("get_requirements", "recall")
builder.add_conditional_edges(
    "recall",
    recaller.recalled_or_new,
    {"new": "get_tests", "recalled": "print_solution"},
)
builder.add_edge("get_tests", "draft_solution")
builder.add_edge("draft_solution", "get_resources")
builder.add_edge("get_resources", "propose_solution")
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
                    There are no heat losses or sources along the rod. Consider the rod one-dimensional. I assume constant initial temperature along the rod.
                """
            )
        ]
    }
)

for m in result["messages"]:
    assert isinstance(m, BaseMessage)
    m.pretty_print()
