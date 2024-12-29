import json
import os
from typing import TypedDict

from IPython.display import Image
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun, WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph as _StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph as _CompiledStateGraph
from langgraph.constants import Send

from tool.logs import get_logger
from tool.models import Solution as _Solution, ResourceForm, EMPTY_RESOURCE
from tool.memory.resource_db import ResourceDB as _ResourceDB, new_custom_database as _resource_db


logger = get_logger()


class ResourceInfo(TypedDict):
    task: str
    context: str
    request: str


_IDENTIFY_SOURCES_PROMPT = """
You are a helpful assistant, that collects for me a knowledge necessary for solving given task.
The task can be either to write a code, do a research or answer a question.
The task can be trivial, in which case I want you to omit the collection of resources.

I will give you the following information:

Task: ...
Context: ...
Solution structure: ...
Already requested resources: ...

You will then write for me a list of new requests for resources, that are not among the Already requested resources.
It is possible that all requests are already written in the Already requested resources.
In such case, there are no resources needed.

It is possible the task requires basic or common knowledge, arithmetics, math or programming.
In such case, there are no resources needed.

Collect as low amount of resources as possible. Do not collect any information, that does not
directly contribute to solving the task.
When you think there are no more resources needed, just return an empty list.

Otherwise, for each item, provide description of the source required in the form "Give me <resource description>.
I expect <form of the result/response>. For example:
[
    "I need to find the definition of the term 'machine learning'. I expect to get it as a plain text.",
    "I need a Pythonic program for calculating the least common divisor. I expect to get it as a Python code snippet."
    ...
]
Do not write anything else except the list.
"""


_RESOURCE_FORM_PROMPT = """
You are a helpful assistant that is focused on deciding the form of a resource.

I will provide you with the following information:
Query: This is the query that I need to find the resource for.

Think carefully about what the query is asking for and what form the resource should be in.
If the query is simply asking for an information unrelated to coding, you should respond with 'text'.

You will respond with the resource in the required form:
- If the resource is a text, you return 'text',
- If the resource is a code, you return 'code'.

Do not write anything else.
"""


_GET_RESOURCE_PROMPT = """
You are a helpful assistant, that helps me to find answer or solution to a question or request.

I will give you the following information:

Request: This is the request for the specific information or resource I need.
Form: This is the form of the resource I need (text or code).

Follow these rules:
- Make the response as short as possible, but provide all the necessary information.
- Return code snippet only if the Form is 'code'.
- Return text if the Form is 'text'.


When resource asks for information, provide the information directly. See the example below.
Example:
    Request: I need to find the speed of light.
    Form: text

    Example of a correct response would be:
        "The speed of light is x m/s."
    Example of an incorrect response, that provides code to generate information instead of the information itself would be: "print(299792458)."
        "```python\nspeed_of_light = x  # meters per second (m/s)\n```"
    Example of an incorrect response, that provides the history of the speed of light instead of the number would be:
        "The speed of light was first measured by Hippolyte Fizeau in 1849 and the value was 315,000 km/s."
"""


_ASSESS_RESOURCE_RELEVANCE_PROMPT = """
You are a helpful assistant that determines if the recalled resource is acceptable.

I will provide to you the following information:

Task: This is the task, which solution requires the resource I am asking for.
Context: This is the context of the task.
Request: This is the request for the specific information or resource I need.
Memory: This is the resource that was recalled from memory.

You will respond in the following way:
Reasoning: Here you provide your thoughts on the relevance of the recalled resource to the task in the given context.
Relevance: 'True' if the solution is nonempty and evaluated as relevant to the task in given context
'False' otherwise.

When assessing relevance, follow these rules:
1) Assess only if the resource contains the requested information.
2) Ignore the factual correctness of information.

Do not write anything else.
"""


class ResourceManager:

    SINGLE_RESOURCE_NODE = "get_resource"

    def __init__(self, db_dir_path: str, openai_model: str = "gpt-4o-mini") -> None:
        self._model = ChatOpenAI(model=openai_model)
        self._form_model = ChatOpenAI(model="gpt-3.5-turbo")
        _wikis = [
            WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(lang=lang))
            for lang in ["en", "de", "fr", "es", "it", "cs", "sk", "pl", "ru", "uk", "ja", "zh"]
        ]
        _external_tools = [DuckDuckGoSearchRun(), *_wikis]  # type: ignore
        self._provider = create_react_agent(self._model, tools=_external_tools)
        self._resource_db = _resource_db(db_dir_path)
        self._construct_graph()

    @property
    def db(self) -> _ResourceDB:
        return self._resource_db

    @property
    def graph(self) -> _CompiledStateGraph:
        """Get the solver's compiled graph."""
        return self._graph

    def invoke(self, solution: _Solution) -> _Solution:
        return self._graph.invoke(solution)

    def print_graph_png(self, path: str, name: str = "resource_manager") -> None:
        with open(os.path.join(path, name.rstrip(".png") + ".png"), "wb") as f:
            f.write(Image(self._graph.get_graph().draw_mermaid_png()).data)

    def get_new_requests_for_resources(self, solution: _Solution) -> list[str]:
        assert isinstance(solution, _Solution), f"Expected Solution, got {type(solution)}"
        task, context = solution.task, solution.context
        requests = solution.resources.keys()
        query = (
            f"Task: {task}\n"
            f"Context: {context}\n"
            f"Solution structure: {solution.structure}\n"
            f"Already requested resources: {requests})"
        )
        messages = [SystemMessage(_IDENTIFY_SOURCES_PROMPT), HumanMessage(query)]
        new_requests = json.loads(str(self._model.invoke(messages).content))
        return new_requests

    def memory_relevance(self, task: str, context: str, request: str, memory: str) -> str:
        query = f"Task: {task}\nContext: {context}\nRequest: {request}\nMemory: {memory}"
        answer = self._model.invoke(
            [SystemMessage(_ASSESS_RESOURCE_RELEVANCE_PROMPT), HumanMessage(query)]
        )
        return str(answer.content)

    def _add_requests(self, solution: _Solution) -> _Solution:
        assert isinstance(solution, _Solution), f"Expected _Solution, got {type(solution)}"
        new_requests = self.get_new_requests_for_resources(solution)
        solution.resources.update({request: EMPTY_RESOURCE for request in new_requests})
        return solution

    def _construct_graph(self) -> None:
        bld = _StateGraph(_Solution)
        bld.add_node("add_requests", self._add_requests, input=_Solution)
        bld.add_node(self.SINGLE_RESOURCE_NODE, self._get_single_resource, input=ResourceInfo)

        bld.add_edge(START, "add_requests")
        bld.add_conditional_edges(
            "add_requests", self._get_resources, [self.SINGLE_RESOURCE_NODE, END]
        )
        bld.add_edge(self.SINGLE_RESOURCE_NODE, END)
        self._graph = bld.compile()

    def _get_resources(self, solution: _Solution) -> list[Send]:
        return [
            Send(
                self.SINGLE_RESOURCE_NODE,
                ResourceInfo(task=solution.task, context=solution.context, request=request),
            )
            for request in solution.resources
            if not solution.resources[request] or solution.resources[request] == EMPTY_RESOURCE
        ]

    def _get_single_resource(self, info: ResourceInfo) -> dict:
        result = self._get_single_memory(info)
        content = result["resources"][info["request"]]
        if not content or content == EMPTY_RESOURCE:
            result = self._get_single_new_resource(info["request"])
        return result

    def _get_single_memory(self, info: ResourceInfo) -> dict:
        memory = self._pick_relevant_recalled_result(info["task"], info["context"], info["request"])
        return {"resources": {info["request"]: memory}}

    def _get_single_new_resource(self, request: str) -> dict:
        query = f"Request: {request}\nForm: {self._resource_form(request)}"
        messages = [SystemMessage(_GET_RESOURCE_PROMPT), HumanMessage(query)]
        response = self._provider.invoke({"messages": messages})["messages"][-1].content
        return {"resources": {request: response}}

    def _resource_form(self, request: str) -> ResourceForm:
        form_query = f"Query: {request}"
        form_messages = [SystemMessage(_RESOURCE_FORM_PROMPT), HumanMessage(form_query)]
        return "code" if "code" in self._form_model.invoke(form_messages).content else "text"

    def _pick_relevant_recalled_result(self, task: str, context: str, request: str) -> str:
        form = self._resource_form(request)
        results = self._resource_db.get(form, context, request)
        for result in results:
            if "True" in self.memory_relevance(task, context, request, result.content):
                return result
        return EMPTY_RESOURCE
