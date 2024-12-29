import os
from typing import Literal

import pydantic
from langgraph.graph import StateGraph as _StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph as _CompiledStateGraph
from langchain_core.messages import AIMessage, HumanMessage
from IPython.display import Image

from tool.models import Solution as _Solution, State as _State
from tool.memory.recaller import Recaller
from tool.proposer.proposer import Compiler
from tool.proposer.resources import ResourceManager
from tool.parsers import task_parser
from tool.requirements.requirements import get_requirements
from tool.test_writer.tests import get_tests
from tool.proposer.structure import draft_solution
from tool.validator import Validator


SOLUTIONS_DIR_NAME = "solutions"
RESOURCES_DIR_NAME = "resources"


class Proposer:

    def __init__(self, memory_path: str) -> None:
        self._memory_path = memory_path
        self._resource_manager = ResourceManager(os.path.join(memory_path, RESOURCES_DIR_NAME))
        self._compiler = Compiler(os.path.join(memory_path, SOLUTIONS_DIR_NAME))
        self._construct_graph()
        assert self._graph is not None

    @property
    def graph(self) -> _CompiledStateGraph:
        """Get the solver's compiled graph."""
        return self._graph

    def invoke(self, task: str) -> AIMessage:
        """Build an ad-hoc graph to solve a task."""
        builder = _StateGraph(_State)
        builder.add_node("parse_task", task_parser)
        builder.add_node("propose", self._graph, input=_Solution)
        builder.add_node("print_solution", self._print_solution, input=_Solution)
        builder.add_edge(START, "parse_task")
        builder.add_edge("parse_task", "propose")
        builder.add_edge("propose", "print_solution")
        builder.add_edge("print_solution", END)
        graph = builder.compile()
        return graph.invoke({"messages": [HumanMessage(content=task)]})["messages"][-1]

    def print_graph_png(self, path: str, name: str = "solver") -> None:
        with open(os.path.join(path, name.rstrip(".png") + ".png"), "wb") as f:
            f.write(Image(self._graph.get_graph().draw_mermaid_png()).data)

    def _construct_graph(self) -> None:
        builder = _StateGraph(_Solution)

        builder.add_node("get_structure", draft_solution, input=_Solution)
        builder.add_node("get_resources", self._resource_manager.graph, input=_Solution)
        builder.add_node("compile_solution", self._compiler.compile, input=_Solution)

        builder.add_edge(START, "get_structure")
        builder.add_edge("get_structure", "get_resources")
        builder.add_edge("get_resources", "compile_solution")
        builder.add_edge("compile_solution", END)
        self._graph = builder.compile()

    def _print_solution(self, solution: _Solution) -> _State:
        return _State(messages=[AIMessage(content=solution.solution)])


class IterativeProposer:

    MAX_ATTEMPTS = 2

    def __init__(self, memory_path: str) -> None:
        self._memory_path = memory_path
        self._proposer = Proposer(memory_path)
        self._validator = Validator()
        self._construct_graph()
        assert self._graph is not None

    @property
    def graph(self) -> _CompiledStateGraph:
        """Get the solver's compiled graph."""
        return self._graph

    def invoke(self, solution: _Solution) -> AIMessage:
        """Build an ad-hoc graph to solve a task."""
        builder = _StateGraph(_Solution)
        builder.add_node("solve", self._graph, input=_Solution)
        builder.add_node("print_solution", self._print_solution, input=_Solution)

        builder.add_edge(START, "solve")
        builder.add_edge("solve", "print_solution")
        builder.add_edge("print_solution", END)

        graph = builder.compile()
        return graph.invoke(
            {"messages": [HumanMessage(content=str(solution.model_dump_json(indent=4)))]}
        )["messages"][-1]

    def print_graph_png(self, path: str, name: str = "solver") -> None:
        with open(os.path.join(path, name.rstrip(".png") + ".png"), "wb") as f:
            f.write(Image(self._graph.get_graph().draw_mermaid_png()).data)

    def _construct_graph(self) -> None:
        builder = _StateGraph(_Solution)

        builder.add_node("input", self._input, input=_Solution)
        builder.add_node("propose", self._proposer.graph, input=_Solution)
        builder.add_node("validate", self._validator.review, input=_Solution)

        builder.add_edge(START, "input")
        builder.add_edge("input", "validate")
        builder.add_conditional_edges("validate", self._retry, {"retry": "propose", "end": END})
        builder.add_edge("propose", "validate")
        self._graph = builder.compile()

    def _failed_test(self, solution: _Solution) -> bool:
        return any(test.result == "fail" for test in solution.tests)

    def _input(self, solution: _Solution) -> _Solution:
        solution.proposal_tries = 1
        return solution

    def _print_solution(self, solution: _Solution) -> _State:
        return _State(messages=[AIMessage(content=solution.solution)])

    def _retry(self, solution: _Solution) -> Literal["retry", "end"]:
        max_attempts = max(self.MAX_ATTEMPTS, 1)
        if self._failed_test(solution):
            print(
                f"Attempt {solution.proposal_tries}/{max_attempts} to propose fully valid solution was unsuccessful. "
                f"{len([test for test in solution.tests if test.result == 'fail'])} out of {len(solution.tests)} tests failed."
            )
            print("\nFailed tests:")
            for t in solution.tests:
                if t.result == "fail":
                    print(f"{t.description}\nCritique: {t.critique_of_last_run}\n")
            if solution.proposal_tries < max_attempts:
                return "retry"
        return "end"


class Solver:

    def __init__(self, memory_path: str) -> None:
        self._recaller = Recaller(os.path.join(memory_path, SOLUTIONS_DIR_NAME))
        self._proposer = Proposer(memory_path)
        self._iterator = IterativeProposer(memory_path)
        self._construct_graph()
        assert self._graph is not None

    @property
    def graph(self) -> _CompiledStateGraph:
        """Get the solver's compiled graph."""
        return self._graph

    def invoke(self, task: str) -> AIMessage:
        """Build an ad-hoc graph to solve a task."""
        builder = _StateGraph(_State)

        builder.add_node("parse_task", task_parser)
        builder.add_node("solve", self._graph, input=_Solution)
        builder.add_node("print_solution", self._print_solution, input=_Solution)

        builder.add_edge(START, "parse_task")
        builder.add_edge("parse_task", "solve")
        builder.add_edge("solve", "print_solution")
        builder.add_edge("print_solution", END)

        graph = builder.compile()
        return graph.invoke({"messages": [HumanMessage(content=task)]})["messages"][-1]

    def print_graph_png(self, path: str, name: str = "solver") -> None:
        with open(os.path.join(path, name.rstrip(".png") + ".png"), "wb") as f:
            f.write(Image(self._graph.get_graph(xray=True).draw_mermaid_png()).data)

    def _construct_graph(self) -> None:
        builder = _StateGraph(_Solution)
        builder.add_node("get_requirements", get_requirements)
        builder.add_node("recall", self._recaller.recall, input=_Solution)
        builder.add_node("add_tests", get_tests, input=_Solution)
        builder.add_node("propose_new", self._proposer.graph, input=_Solution)
        builder.add_node("validate_and_improve", self._iterator.graph, input=_Solution)

        builder.add_edge(START, "get_requirements")
        builder.add_edge("get_requirements", "recall")
        builder.add_conditional_edges(
            "recall",
            self._recaller.recalled_or_new,
            {"new": "add_tests", "recalled": "validate_and_improve"},
        )
        builder.add_edge("add_tests", "propose_new")
        builder.add_edge("propose_new", "validate_and_improve")
        builder.add_edge("validate_and_improve", END)
        self._graph = builder.compile()

    def _print_solution(self, solution: _Solution) -> _State:
        return _State(messages=[AIMessage(content=solution.solution)])
