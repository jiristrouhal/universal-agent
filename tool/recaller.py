from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import START, END
from langchain_openai import ChatOpenAI

from tool.models import TaskPlain, TaskWithSolutionRecall
from tool.solution_db import SolutionDB


SOLUTION_JUDGE_PROMPT = """
You are a helpful assistant, that analyzes the user's request and provides the most relevant solutions from memory.

Look on the given task asked in given context and assess the best solution from the list of solutions to similar problems from the past.
It is possible none of them are relevant, in that case, you respond with a message that there is no relevant solution.

Task:\n{task}
Context:\n{context}
Recalled solutions:\n{solutions}

Your answer must be in the following format:

Reasoning: Here you think step by step about each of the solutions with respect to the task and context.
Best solution: Here you choose the best solution. Do not write anything else.
"""


_db = SolutionDB()
_model = ChatOpenAI(model="gpt-4o-mini")


def recall(task: TaskWithSolutionRecall) -> TaskWithSolutionRecall:
    task_str = f"Task is to {task.task} in the context of {task.context}."
    solutions = [content for content, _ in _db.search(task_str, k=3)]
    for k in range(len(solutions)):
        solutions[k] = f"{k + 1}. {solutions[k][0]}"
    solutions_str = "\n\t".join(solutions)
    result = _assess_solutions(task.task, task.context, solutions_str)
    task_with_solution_recall = TaskWithSolutionRecall(
        task=task.task, context=task.context, solution_recall=result
    )
    return task_with_solution_recall


def _assess_solutions(task: str, context: str, solutions: str) -> str:
    formatted_system_prompt = SOLUTION_JUDGE_PROMPT.format(
        task=task, context=context, solutions=solutions
    )
    messages = [SystemMessage(content=formatted_system_prompt)]
    result = str(_model.invoke(messages).content)
    return result


SOLUTION_OK_PROMPT = """
You are a helpful assistant that determines if the solution provided is acceptable.

I will provide to you the following information:

Task: ...
Context: ...
Solution recall: ...

You will respond with 'True' if the solution is nonempty and evaluated as valid, 'False' otherwise. Do not write anything else.
"""


def recalled_or_new(
    task_with_solution_recall: TaskWithSolutionRecall,
) -> Literal["recalled", "new"]:
    task = task_with_solution_recall
    result = _model.invoke(
        [
            SystemMessage(content=SOLUTION_OK_PROMPT),
            HumanMessage(
                content=f"Task: {task.task}\nContext: {task.context}\nSolution recall: {task.solution_recall}"
            ),
        ]
    )
    if "True" in result.content:
        return "recalled"
    return "new"
