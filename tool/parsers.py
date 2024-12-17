from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from trustcall import create_extractor

from tool.models import Solution as Solution, State as _State


model = ChatOpenAI(name="gpt-4o-mini")
task_extractor = create_extractor(model, tools=[Solution], tool_choice="Solution")
TASK_EXTRACTOR_PROMPT = (
    "Extract a task context and the required form of solution from the following messages."
)


def task_parser(state: _State) -> Solution:
    """This function extracts a task from the messages in the state and returns it as a single message
    containing the task - an empty solution.
    """
    messages = [SystemMessage(TASK_EXTRACTOR_PROMPT)] + state["messages"]  # type: ignore
    result: Solution = task_extractor.invoke({"messages": messages})["responses"][0]
    return Solution(task=result.task, context=result.context, form=result.form)
