from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from models import Task, State
from solution_db import SolutionDB


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


def recall(task: Task) -> State:
    task_str = f"Task is to {task.task} in the context of {task.context}."
    solutions = _db.search(task_str, k=3)
    for k in range(len(solutions)):
        solutions[k] = f"{k + 1}. {solutions[k][0]}"
    solutions_str = "\n\t".join(solutions)
    result = _assess_solutions(task.task, task.context, solutions_str)
    return State(messages=[AIMessage(content=result)])


def _assess_solutions(task: str, context: str, solutions: str) -> str:
    formatted_system_prompt = SOLUTION_JUDGE_PROMPT.format(
        task=task, context=context, solutions=solutions
    )
    messages = [SystemMessage(content=formatted_system_prompt)]
    result = str(_model.invoke(messages).content)
    return result
