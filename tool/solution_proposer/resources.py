import json

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun, WikipediaAPIWrapper

from tool.models import TaskWithSources, TaskWithSolutionStructure, State, ResourceForm, Resource
from tool.memory.resource_db import (
    ResourceDB as _ResourceDB,
    new_custom_database as _new_custom_db,
    database as _default_database,
)


_database = _default_database


_IDENTIFY_SOURCES_PROMPT = """
You are a helpful assistant, that helps me to design solution to a task.

I will give you the following information:

Task: ...
Context: ...
Solution draft: ...

Please, identify the sources of information that you used to design the solution. Respond in a form of list.
It is possible, that for very simple solutions, there are no sources required.
For each item, provide detailed description of the source required in the form "Give me [resource description]. I expect to get it as [form of the result/response].

For example:
[
    "I need to find the definition of the term 'machine learning'. I expect to get it as a plain text.",
    "I need a Pythonic program for calculating the least common divisor. I expect to get it as a Python code snippet."
    ...
]

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


_EXTRACT_RELEVANT_RESOURCE_PART_PROMPT = """
You are a helpful assistant, that helps me to find answer or solution to a question or request.


I will give you the following information:
Task: This is the task, which solution requires the information I am asking for.
Context: This is the context of the task.
Resource: This is the text, which contains the information I need to extract.

Please, extract the relevant part of the resource, which contains the information I need to solve the task.

If task requires a code, that is par of the resource, extract the code without changes.
If task requires a text, retain only the relevant part of the text or summarize it, so the output is at maximum 6 sentences long.
"""


_model = ChatOpenAI(model="gpt-4o-mini")
_tools = [WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())]  # type: ignore
_resource_provider = create_react_agent(_model, tools=_tools)


def collect_resources(draft: TaskWithSolutionStructure) -> TaskWithSources:
    query = (
        f"Task: {draft.task}\n"
        f"Context: {draft.context}\n"
        f"Solution draft: {draft.solution_structure}"
    )
    extraction_messages = [
        SystemMessage(content=_IDENTIFY_SOURCES_PROMPT),
        HumanMessage(content=query),
    ]
    requests_for_sources: list[str] = list(
        json.loads(str(_model.invoke(extraction_messages).content))
    )
    requested_sources = dict.fromkeys(requests_for_sources, "Not provided")

    for request in requested_sources:
        results = _database.get(form="text", context=draft.context, request=request)
        for r in results:
            answer = _model.invoke(
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
            print(f"Resource for '{request}' already provided")
            continue
        full_request = f"Task: {draft.task}\nContext: {draft.context}\nRequest: {request}"
        messages = [SystemMessage(content=_GET_RESOURCE_PROMPT), HumanMessage(content=full_request)]
        result = _resource_provider.invoke({"messages": messages})
        extraction_query = f"Task: {draft.task}\nContext: {draft.context}\nResource: {str(result)}"
        extraction_messages = [
            SystemMessage(content=_EXTRACT_RELEVANT_RESOURCE_PART_PROMPT),
            HumanMessage(content=extraction_query),
        ]
        extracted_result = str(_model.invoke(extraction_messages).content)
        _database.add(
            Resource(form="text", context=draft.context, request=request, content=extracted_result)
        )
        requested_sources[request] = extracted_result

    return TaskWithSources(
        task=draft.task,
        context=draft.context,
        requirements=draft.requirements,
        tests=draft.tests,
        solution_structure=draft.solution_structure,
        sources=requested_sources,
    )


def new_custom_database(db_location: str = "") -> None:
    global _database
    _database = _new_custom_db(db_location)


def get_database() -> _ResourceDB:
    return _database


def print_sources(draft_with_sources: TaskWithSources) -> State:
    return State(messages=[HumanMessage(content=str(draft_with_sources.model_dump_json(indent=4)))])
