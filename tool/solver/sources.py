import json

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun, WikipediaAPIWrapper
from tool.models import TaskWithSources, TaskWithSolutionStructure, State


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


_model = ChatOpenAI(model="gpt-4o-mini")
_tools = [WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())]  # type: ignore
_resource_provider = create_react_agent(_model, tools=_tools)


def collect_sources(draft: TaskWithSolutionStructure) -> TaskWithSources:
    query = (
        f"Task: {draft.task}\n"
        f"Context: {draft.context}\n"
        f"Solution draft: {draft.solution_structure}"
    )
    messages = [SystemMessage(content=_IDENTIFY_SOURCES_PROMPT), HumanMessage(content=query)]
    requests_for_sources: list[str] = list(json.loads(str(_model.invoke(messages).content)))
    requested_sources = dict.fromkeys(requests_for_sources, "Not provided")
    for request in requested_sources:
        full_request = f"Task: {draft.task}\n" f"Context: {draft.context}\n" f"Request: {request}"
        messages = [SystemMessage(content=_GET_RESOURCE_PROMPT), HumanMessage(content=full_request)]
        result = _resource_provider.invoke({"messages": messages})
        requested_sources[request] = str(result)
    return TaskWithSources(
        task=draft.task,
        context=draft.context,
        requirements=draft.requirements,
        tests=draft.tests,
        solution_structure=draft.solution_structure,
        sources=requested_sources,
    )


def print_sources(draft_with_sources: TaskWithSources) -> State:
    return State(messages=[HumanMessage(content=str(draft_with_sources.model_dump_json(indent=4)))])
