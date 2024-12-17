import json

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun, WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun

from tool.models import Solution, Solution, ResourceForm, EMPTY_RESOURCE, Resource as _Resource
from tool.memory.resource_db import ResourceDB as _ResourceDB, new_custom_database as _resource_db


_IDENTIFY_SOURCES_PROMPT = """
You are a helpful assistant, that helps me to design solution to a task.

I will give you the following information:

Task: ...
Context: ...
Solution draft: ...
Existing requests for sources ...

Identify the resources of information that you need to design the solution. Respond in a form of list, for example:
[
    "I need to find the definition of the term 'machine learning'. I expect to get it as a plain text.",
    "I need a Pythonic program for calculating the least common divisor. I expect to get it as a Python code snippet."
    ...
]

Write only requests, that ARE NOT included in the Existing requests for sources.
Please, follow these instructions:

1) It is possible, that for very simple solutions) (simple arithmetic operations), there are no resources required.
2) For each item, provide detailed description of the source required in the form "Give me [resource description]. I expect to get it as [form of the result/response].
3) Do not provide any additional information. Do not write anything else.
"""


_RESOURCE_FORM_PROMPT = """
You are a helpful assistant that helps me to find the resource in the required form.

I will provide you with the following information:
Query: This is the query that I need to find the resource for.

You will respond with the resource in the required form:
- If the resource is a text, you return 'text',
- If the resource is a code, you return 'code'.

Do not write anything else.
"""


_GET_RESOURCE_PROMPT = """
You are a helpful assistant, that helps me to find answer or solution to a question or request.

I will give you the following information:

Task: This is the task, which solution requires the information I am asking for.
Context: This is the context of the task.
Request: This is the request for the specific information or resource I need.
"""


_ASSESS_RECALLED_RESOURCES_PROMPT = """
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

    def __init__(self, db_dir_path: str, openai_model: str = "gpt-4o-mini") -> None:
        self._model = ChatOpenAI(model=openai_model)
        _external_tools = [  # type: ignore
            DuckDuckGoSearchRun(),
            WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),
        ]
        self._provider = create_react_agent(self._model, tools=_external_tools)
        self._resource_db = _resource_db(db_dir_path)

    @property
    def db(self) -> _ResourceDB:
        return self._resource_db

    def get_resources(self, solution: Solution, use_external: bool = True) -> Solution:
        task, context = solution.task, solution.context
        query = (
            f"Task: {task}\n"
            f"Context: {context}\n"
            f"Solution draft: {solution.solution_structure}\n"
            f"Existing requests for sources: {list(solution.resources.keys())}"
        )
        messages = [SystemMessage(_IDENTIFY_SOURCES_PROMPT), HumanMessage(query)]
        requests_for_sources: list[str] = list(solution.resources.keys()) + list(
            json.loads(str(self._model.invoke(messages).content))
        )
        requested_resources = dict.fromkeys(requests_for_sources, EMPTY_RESOURCE)
        self._recall_resources_from_memory(task, context, requested_resources)
        self._get_new_requested_resources(task, context, requested_resources, use_external)
        solution.resources.update(requested_resources)
        return solution

    def _get_new_requested_resources(
        self,
        task: str,
        context: str,
        requested_resources: dict[str, str],
        use_external: bool = True,
    ) -> None:
        for request in requested_resources:
            if requested_resources[request] != EMPTY_RESOURCE:
                continue
            full_request = f"Task: {task}\nContext: {context}\nRequest: {request}"
            messages = [
                SystemMessage(content=_GET_RESOURCE_PROMPT),
                HumanMessage(content=full_request),
            ]
            if use_external:
                result = self._provider.invoke({"messages": messages})["messages"][-1].content
            else:
                result = self._model.invoke(messages).content
            requested_resources[request] = str(result)

            form = self._get_resource_form(request)
            self._resource_db.add(
                _Resource(form=form, context=context, request=request, content=result)
            )

    def memory_relevance(self, task: str, context: str, request: str, memory: str) -> str:
        query = f"Task: {task}\nContext: {context}\nRequest: {request}\nMemory: {memory}"
        answer = self._model.invoke(
            [
                SystemMessage(content=_ASSESS_RECALLED_RESOURCES_PROMPT),
                HumanMessage(content=query),
            ]
        ).content
        return str(answer)

    def _get_resource_form(self, request: str) -> ResourceForm:
        form_query = f"Query: {request}"
        form_messages = [SystemMessage(_RESOURCE_FORM_PROMPT), HumanMessage(form_query)]
        return "code" if "code" in self._model.invoke(form_messages).content else "text"

    def _recall_resources_from_memory(
        self,
        task: str,
        context: str,
        requested_resources: dict[str, str],
    ) -> None:
        for request in requested_resources:
            form = self._get_resource_form((request))
            results = self._resource_db.get(form=form, context=context, request=request)
            for result in results:
                if "True" in self.memory_relevance(task, context, request, result.content):
                    print("Recalled resource is acceptable")
                    requested_resources[request] = result.content
                    break
