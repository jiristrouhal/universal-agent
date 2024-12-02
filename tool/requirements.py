from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from models import TaskToSolve, TaskWithSolutionRecall, State


SOLUTION_REQUIREMENT_PROMPT = """
You are a helpful assistant, that analyzes task to be solved and thinks about requirements on the solution.

I will give you the following information:
Task: ...
Context: ...

You will responds with the list of requirements that the solution must meet.
1. Requirement 1
2. Requirement 2
...

Do not write anything else.
"""

_model = ChatOpenAI(model="gpt-4o-mini")


def get_requirements(task_with_recall: TaskWithSolutionRecall) -> TaskToSolve:
    task_str = f"Task: {task_with_recall.task}\nContext: {task_with_recall.context}"
    messages = [SystemMessage(content=SOLUTION_REQUIREMENT_PROMPT), HumanMessage(content=task_str)]
    result = _model.invoke(messages)
    return TaskToSolve(
        task=task_with_recall.task, context=task_with_recall.context, requirements=result.content
    )


def print_requirements(task: TaskToSolve) -> State:
    return State(messages=[SystemMessage(content=task.requirements)])
