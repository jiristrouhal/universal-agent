from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from trustcall import create_extractor

from tool.models import TaskPlain, State


model = ChatOpenAI(name="gpt-4o-mini")
task_extractor = create_extractor(model, tools=[TaskPlain], tool_choice="TaskPlain")
TASK_EXTRACTOR_PROMPT = "Extract a task with context from the following messages."


def parse_task(state: State) -> TaskPlain:
    """This function extracts a task from the messages in the state and returns it as a single message
    containing the task.
    """
    messages = [SystemMessage(TASK_EXTRACTOR_PROMPT)]
    messages.extend(state["messages"])  # type: ignore
    result = task_extractor.invoke({"messages": messages})["responses"][0]
    assert isinstance(result, TaskPlain)
    return result


def present_task(task: TaskPlain) -> State:
    """This function presents the task to the user."""
    return State(messages=[SystemMessage(f"Task: {task.task}\nContext: {task.context}")])
