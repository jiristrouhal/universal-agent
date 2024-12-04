from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from tool.models import TaskWithSourceAugmentedDraft, TaskWithSolution, State


_PROPOSE_SOLUTION_PROMPT = """
You are an experienced problem solver, that helps me to design solution to a task. I will provide you with a following information:

Context: ...
Task: ...
Tests: ...
Solution structure: ...
Resources: ...

Please, follow these guidelines:
1) The solution must be relevant and testable in the context of the task.
2) Use only the provided Resources for building the solution. Do not make any assumptions or ad-hoc information retrievals.
3) All points from the Solution structure must be addressed in the solution.
4) The solution must pass all the Tests.
5) Write only the solution, helper functions and imports. Do not write any additional code.

You should respond to me only with the solution. Do not write anything else.
"""

_model = ChatOpenAI(model="gpt-4o-mini")


def propose_solution(draft: TaskWithSourceAugmentedDraft) -> TaskWithSolution:
    """Propose a solution to the task and store it in the solution database."""
    query = (
        f"Context: {draft.context}\n"
        f"Task: {draft.task}\n"
        f"Tests: {draft.tests}\n"
        f"Solution structure: {draft.solution_draft}"
    )
    messages = [SystemMessage(content=_PROPOSE_SOLUTION_PROMPT), HumanMessage(content=query)]
    solution = str(_model.invoke(messages).content)
    return TaskWithSolution(
        task=draft.task, context=draft.context, tests=draft.tests, solution=solution
    )


def print_solution(solution: TaskWithSolution) -> State:
    return State(messages=[AIMessage(content=solution.solution)])
