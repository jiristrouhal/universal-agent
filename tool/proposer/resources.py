import json

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun, WikipediaAPIWrapper

from tool.models import Solution, Solution, ResourceForm
from tool.memory.resource_db import ResourceDB as _ResourceDB, new_custom_database as _resource_db


_IDENTIFY_SOURCES_PROMPT = """
You are a helpful assistant, that helps me to design solution to a task.

I will give you the following information:

Task: ...
Context: ...
Solution draft: ...

Please, identify the resources of information that you used to design the solution. Respond in a form of list.
It is possible, that for very simple solutions, there are no resources required.
For each item, provide detailed description of the source required in the form "Give me [resource description]. I expect to get it as [form of the result/response].

For example:
[
    "I need to find the definition of the term 'machine learning'. I expect to get it as a plain text.",
    "I need a Pythonic program for calculating the least common divisor. I expect to get it as a Python code snippet."
    ...
]

Do not write anything else.
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
Recalled resource: This is the resource that was recalled from memory.

You will respond with 'True' if the solution is nonempty and evaluated as valid, 'False' otherwise. Do not write anything else.
"""


class ResourceManager:

    def __init__(self, db_dir_path: str, openai_model: str = "gpt-4o-mini") -> None:
        self._model = ChatOpenAI(model=openai_model)
        _tools = [WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())]  # type: ignore
        self._resource_provider = create_react_agent(self._model, tools=_tools)
        self._resource_db = _resource_db(db_dir_path)

    @property
    def db(self) -> _ResourceDB:
        return self._resource_db

    def get_resources(self, draft: Solution) -> Solution:
        query = (
            f"Task: {draft.task}\n"
            f"Context: {draft.context}\n"
            f"Solution draft: {draft.solution_structure}"
        )
        messages = [SystemMessage(_IDENTIFY_SOURCES_PROMPT), HumanMessage(query)]
        requests_for_sources: list[str] = list(
            json.loads(str(self._model.invoke(messages).content))
        )

        # Get the form of solution - text or code
        form_query = f"Query: {requests_for_sources[0]}"
        form_messages = [SystemMessage(_RESOURCE_FORM_PROMPT), HumanMessage(form_query)]
        form: ResourceForm = (
            "code" if "code" in self._model.invoke(form_messages).content else "text"
        )

        requested_sources = dict.fromkeys(requests_for_sources, "Not provided")
        for request in requested_sources:
            results = self._resource_db.get(form=form, context=draft.context, request=request)
            for r in results:
                answer = self._model.invoke(
                    [
                        SystemMessage(content=_ASSESS_RECALLED_RESOURCES_PROMPT),
                        HumanMessage(
                            content=f"Task: {draft.task}\nContext: {draft.context}\nSolution recall: {r.content}"
                        ),
                    ]
                ).content
                if "True" in str(answer):
                    print("Recalled resource is acceptable")
                    requested_sources[request] = r.content
                    break

        for request in requested_sources:
            if requested_sources[request] != "Not provided":
                continue
            full_request = f"Task: {draft.task}\nContext: {draft.context}\nRequest: {request}"
            messages = [
                SystemMessage(content=_GET_RESOURCE_PROMPT),
                HumanMessage(content=full_request),
            ]
            result = self._resource_provider.invoke({"messages": messages})["messages"][-1].content
            requested_sources[request] = str(result)

        return Solution(
            task=draft.task,
            context=draft.context,
            requirements=draft.requirements,
            tests=draft.tests,
            solution_structure=draft.solution_structure,
            resources=requested_sources,
        )
