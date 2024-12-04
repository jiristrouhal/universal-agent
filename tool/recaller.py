from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from tool.models import TaskWithSolutionRecall
from tool.memory.solution_db import database, Solution


SOLUTION_JUDGE_PROMPT = """
You are a helpful assistant, that analyzes the user's request and provides the most relevant solutions from memory.

Look on the given task asked in given context and assess the best solution from the list of solutions to similar problems from the past.
It is possible none of them are relevant, in that case, you respond with a message that there is no relevant solution.

Task:\n{task}
Context:\n{context}
Recalled solutions:\n{solutions}

Your answer must be in the following format:

Reasoning: Here you think step by step about Recalled solutions with respect to the task and context.
Best solution: Here you choose the best solution. Do not write anything else. It is possible none of the solutions are relevant.
"""


SOLUTION_OK_PROMPT = """
You are a helpful assistant that determines if the solution provided is acceptable.

I will provide to you the following information:

Task: ...
Context: ...
Solution recall: ...

You will respond with 'True' if the solution is nonempty and evaluated as valid, 'False' otherwise. Do not write anything else.
"""


_model = ChatOpenAI(model="gpt-4o-mini")


def recall(task: TaskWithSolutionRecall) -> TaskWithSolutionRecall:
    recalled_solutions = database.get_solutions(task.context, task.task, k=3)
    solutions_str = ""
    for k in range(len(recalled_solutions)):
        solutions_str += f"Solution {k}:\n" + _recalled_solution_description(recalled_solutions[k])
        if k < len(recalled_solutions) - 1:
            solutions_str += "\n\n"
    result = _assess_solutions(task.task, task.context, solutions_str)
    task_with_solution_recall = TaskWithSolutionRecall(
        task=task.task, context=task.context, solution_recall=result
    )
    return task_with_solution_recall


def _recalled_solution_description(solution: Solution) -> str:
    tests_str = "\n\t".join(
        [f"{test.test}\n\t{test.critique_of_last_run}" for test in solution.tests]
    )
    return f"Task: {solution.task}\nContext: {solution.context}\nSolution:\n{solution.solution}\n"


def _assess_solutions(task: str, context: str, solutions: str) -> str:
    formatted_system_prompt = SOLUTION_JUDGE_PROMPT.format(
        task=task, context=context, solutions=solutions
    )
    messages = [SystemMessage(content=formatted_system_prompt)]
    result = str(_model.invoke(messages).content)
    return result


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
