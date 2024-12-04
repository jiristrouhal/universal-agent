import json

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from tool.models import TaskToSolve, TaskWithSolutionRecall


SOLUTION_REQUIREMENT_PROMPT = """
You are a helpful assistant, that analyzes task to be solved and thinks about requirements on the solution.

I will give you the following information:
Task: ...
Context: ...

You will responds with the pythonic list of requirements that the solution must meet:
["requirement1", "requirement2", ...]

Do not write anything else.
"""

_model = ChatOpenAI(model="gpt-4o-mini")


def get_requirements(task_with_recall: TaskWithSolutionRecall) -> TaskToSolve:
    task_str = f"Task: {task_with_recall.task}\nContext: {task_with_recall.context}"
    messages = [SystemMessage(content=SOLUTION_REQUIREMENT_PROMPT), HumanMessage(content=task_str)]
    result = _model.invoke(messages)
    return TaskToSolve(
        task=task_with_recall.task,
        context=task_with_recall.context,
        requirements=list(json.loads(str(result.content))),
    )
