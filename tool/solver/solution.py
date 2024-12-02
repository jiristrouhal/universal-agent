from pprint import pprint

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from tool.models import TaskWithTests, TaskWithSolutionDraft, State


SOLUTION_PROMPT = """
You are a helpful assistant, that helps me to design solution to a task.

I will give you the following information:

Task: ...
Context: ...
Requirements: ...
Tests: ...

Think step by step about the solution structure and write it down. Identify all necessary resources (functions, information etc.). Do not mention anything else.
Please, respond in the following format:

Step-by-step structure of the solution:
...
Required sources:
...
"""


_model = ChatOpenAI(model="gpt-4o-mini")


def propose_solution(task_with_tests: TaskWithTests) -> TaskWithSolutionDraft:
    task_str = (
        f"Task: {task_with_tests.task}\n"
        f"Context: {task_with_tests.context}\n"
        f"Requirements: {task_with_tests.requirements}\n"
        f"Tests: {task_with_tests.tests}"
    )
    messages = [SystemMessage(content=SOLUTION_PROMPT), HumanMessage(content=task_str)]
    result = _model.invoke(messages)
    return TaskWithSolutionDraft(
        task=task_with_tests.task,
        context=task_with_tests.context,
        requirements=task_with_tests.requirements,
        tests=task_with_tests.tests,
        solution_draft=str(result.content),
    )


def print_solution_draft(task_with_solution_draft: TaskWithSolutionDraft) -> State:
    return State(
        messages=[HumanMessage(content=str(task_with_solution_draft.model_dump_json(indent=4)))]
    )
