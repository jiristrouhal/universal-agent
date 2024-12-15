import os

from IPython.display import Image
from langgraph.graph import StateGraph as _StateGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain.tools import Tool

from tool.solver import Solver
from tool.models import State as _State, Solution as _Solution
from tool.validator import Validator
from tool.parsers import task_parser


CHAT_MODEL = "gpt-4o-mini"
CHAT_PROMPT = """
You are a helpful and brief assistant, that accepts helps me to formulate a task and when it is formulated, you call a tool to provide solution.

Think about my input and ask me for any additional info if necessary. Call a tool if the task is more complex, includes specific knowledge or
requires running and testing code.

Respond briefly.
"""


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

    def chat(self) -> None:
        chat = ChatOpenAI(model=CHAT_MODEL)
        tool = Tool(
            "think-hard",
            self.invoke,
            description=(
                "This method accepts a task, thinks about the solution and returns it. It is recommended to include both task and a context."
                "Formulate task as a plain text."
            ),
        )
        chat_with_tools = chat.bind_tools([tool])
        messages: list[BaseMessage] = [SystemMessage(CHAT_PROMPT)]
        while True:
            user_message = input("Me: ")
            if not user_message.strip():
                continue
            messages.append(HumanMessage(content=user_message))
            response = chat_with_tools.invoke(messages)
            print("Tool:", str(response.content))
            messages.append(response)

    def invoke(self, task: str) -> AIMessage:
        print(f"Processing task: {task}.")
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
