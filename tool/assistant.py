import os
from typing import Literal

from IPython.display import Image
from langgraph.graph import StateGraph as _StateGraph, START, END
from langgraph.graph.state import CompiledGraph
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

from tool.solver import Solver
from tool.models import State as _State, Solution as _Solution
from tool.parsers import task_parser
from tool.memory.resource_db import ResourceDB as _ResourceDB
from tool.memory.solution_db import SolutionDB as _SolutionDB


CHAT_MODEL = "gpt-4o-mini"
CHAT_PROMPT = """
You are a helpful and brief assistant, that helps me to formulate a task and when it is formulated, you call a tool to provide solution.

Think about my input and ask me for any additional info if necessary. Call a tool after the task is well-formulated.
requires running and testing code.
"""


class Assistant:
    def __init__(self, memory_path):
        self.solver = Solver(memory_path)
        self._compile_graph()

    @property
    def resource_db(self) -> _ResourceDB:
        return self.solver.resource_db

    @property
    def solution_db(self) -> _SolutionDB:
        return self.solver.solution_db

    def invoke_in_loop(self, task: str = "") -> AIMessage:
        while True:
            if not task.strip():
                task = input("Please, write me a task: ")
            else:
                result = self.invoke(task).content
                print(result)
                task = ""

    def invoke(self, task: str) -> AIMessage:
        """This method accepts a task, thinks about the solution and returns it.

        It is recommended to include both task and a context. Formulate task as a plain text.
        """
        print(f"Processing task: {task}")
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

    def _compile_graph(self) -> None:
        builder = _StateGraph(_Solution)
        builder.add_node("solver", self.solver.graph, input=_Solution)
        builder.add_edge(START, "solver")
        builder.add_edge("solver", END)
        self._graph = builder.compile()

    def _invoke_and_append_result(
        self, graph: CompiledGraph, messages: list[BaseMessage], human_message: str
    ) -> None:
        messages.append(HumanMessage(content=human_message))
        response: BaseMessage = graph.invoke({"messages": messages})["messages"][-1]
        print("Tool:", str(response.content))
        messages.append(response)

    def print_graph_png(self, path: str, name: str = "assistant") -> None:
        with open(os.path.join(path, name.rstrip(".png") + ".png"), "wb") as f:
            f.write(Image(self._graph.get_graph(xray=True).draw_mermaid_png()).data)

    def _output_solution(self, solution: _Solution) -> _State:
        return _State(messages=[AIMessage(content=solution.solution)])
