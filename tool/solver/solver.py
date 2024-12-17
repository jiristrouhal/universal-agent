import os

from langgraph.graph import StateGraph as _StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph as _CompiledStateGraph
from langchain_core.messages import AIMessage, HumanMessage
from IPython.display import Image

from tool.models import Solution as Solution, State as _State
from tool.memory.recaller import Recaller
from tool.proposer.proposer import Proposer
from tool.proposer.resources import ResourceManager
from tool.parsers import task_parser
from tool.proposer.requirements import get_requirements
from tool.proposer.tests import get_tests
from tool.proposer.structure import draft_solution


SOLUTIONS_DIR_NAME = "solutions"
RESOURCES_DIR_NAME = "resources"


class Solver:

    __slots__ = ["_memory_path", "_recaller", "_resource_manager", "_proposer", "_graph"]

    def __init__(self, memory_path: str) -> None:
        self._memory_path = memory_path
        self._recaller = Recaller(os.path.join(memory_path, SOLUTIONS_DIR_NAME))
        self._resource_manager = ResourceManager(os.path.join(memory_path, RESOURCES_DIR_NAME))
        self._proposer = Proposer(os.path.join(memory_path, SOLUTIONS_DIR_NAME))
        self._construct_graph()
        assert self._graph is not None

    @property
    def graph(self) -> _CompiledStateGraph:
        """Get the solver's compiled graph."""
        return self._graph

    @property
    def proposer(self) -> Proposer:
        return self._proposer

    def invoke(self, task: str) -> AIMessage:
        """Build an ad-hoc graph to solve a task."""
        builder = _StateGraph(_State)
        builder.add_node("parse_task", task_parser)
        builder.add_node("solve", self._graph, input=Solution)
        builder.add_node("print_solution", self._print_solution, input=Solution)
        builder.add_edge(START, "parse_task")
        builder.add_edge("parse_task", "solve")
        builder.add_edge("solve", "print_solution")
        builder.add_edge("print_solution", END)
        graph = builder.compile()
        return graph.invoke({"messages": [HumanMessage(content=task)]})["messages"][-1]

    def print_graph_png(self, path: str, name: str = "solver") -> None:
        with open(os.path.join(path, name.rstrip(".png") + ".png"), "wb") as f:
            f.write(Image(self._graph.get_graph().draw_mermaid_png()).data)

    def _construct_graph(self) -> None:
        builder = _StateGraph(Solution)
        builder.add_node("get_requirements", get_requirements)
        builder.add_edge(START, "get_requirements")
        builder.add_edge("get_requirements", "recall")
        builder.add_node("recall", self._recaller.recall, input=Solution)
        builder.add_node("get_tests", get_tests, input=Solution)
        builder.add_node("get_structure", draft_solution, input=Solution)
        builder.add_node("get_resources", self._resource_manager.get_resources, input=Solution)
        builder.add_node("propose_solution", self._proposer.propose_solution, input=Solution)

        builder.add_edge(START, "get_requirements")
        builder.add_edge("get_requirements", "recall")
        builder.add_conditional_edges(
            "recall",
            self._recaller.recalled_or_new,
            {"new": "get_tests", "recalled": END},
        )
        builder.add_edge("get_tests", "get_structure")
        builder.add_edge("get_structure", "get_resources")
        builder.add_edge("get_resources", "propose_solution")
        builder.add_edge("propose_solution", END)
        self._graph = builder.compile()

    def _print_solution(self, solution: Solution) -> _State:
        return _State(messages=[AIMessage(content=solution.solution)])
