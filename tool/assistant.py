import os

from IPython.display import Image
from langgraph.graph import StateGraph as _StateGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage

from tool.solver import Solver
from tool.models import State as _State, Solution as _Solution
from tool.validator import Validator
from tool.parsers import task_parser


class Assistant:
    def __init__(self, memory_path):
        self.solver = Solver(memory_path)
        self.validator = Validator()
        self._compile_graph()

    def _compile_graph(self) -> None:
        builder = _StateGraph(_Solution)
        builder.add_node("solve", self.solver.graph, input=_Solution)
        builder.add_node("validate", self.validator.review, input=_Solution)

        builder.add_edge(START, "solve")
        builder.add_edge("solve", "validate")
        builder.add_edge("validate", END)
        self._graph = builder.compile()

    def invoke(self, task: str) -> AIMessage:
        builder = _StateGraph(_State)
        builder.add_node("parse_task", task_parser, input=_State)
        builder.add_node("core", self._graph, input=_Solution)
        builder.add_node("output_solution", self._output_solution, input=_Solution)
        builder.add_edge(START, "parse_task")
        builder.add_edge("parse_task", "core")
        builder.add_edge("core", "output_solution")
        builder.add_edge("output_solution", END)
        graph = builder.compile()
        return graph.invoke(_State(messages=[HumanMessage(content=task)]))["messages"][-1]

    def print_graph_png(self, path: str, name: str = "solver") -> None:
        with open(os.path.join(path, name.rstrip(".png") + ".png"), "wb") as f:
            f.write(Image(self._graph.get_graph(xray=True).draw_mermaid_png()).data)

    def _output_solution(self, solution: _Solution) -> _State:
        return _State(messages=[AIMessage(content=solution.solution)])
