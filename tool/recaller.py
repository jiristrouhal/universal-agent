from __future__ import annotations
from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from tool.models import TaskWithSolutionRecall as _TaskWithSolutionRecall, Solution as _Solution
from tool.memory.solution_db import get_solution_database as _get_database


SOLUTION_RECALL_PROMPT = """
You are a helpful assistant, that analyzes the user's request and provides the most relevant solutions from memory.

Look on the given task asked in given context and assess the best solution from the list of solutions to similar problems from the past.
It is possible none of them are relevant, in that case, you respond with a message that there is no relevant solution.

Task:\n{task}
Context:\n{context}
Recalled solutions:\n{solutions}

Your answer must be in the following format:

Reasoning: Here you think step by step about Recalled solutions with respect to the task and context.
Best solution: Here you choose the best solution. Do not write anything else. It is possible that none of the solutions is relevant.
"""


SOLUTION_OK_PROMPT = """
You are a helpful assistant that determines if the solution provided is acceptable.

I will provide to you the following information:

Task: ...
Context: ...
Solution recall: ...

You will respond with 'True' if the solution is nonempty and evaluated as valid, 'False' otherwise. Do not write anything else.
"""


class Recaller:

    def __init__(self, db_dir_path: str, openai_model: str = "gpt-4o-mini") -> None:
        self._db = _get_database(db_dir_path)
        self._model = ChatOpenAI(model=openai_model)

    def recall(self, task: _TaskWithSolutionRecall) -> _TaskWithSolutionRecall:
        """Recall the solution from the memory and assess if it is relevant."""
        recalled_solutions = self._db.get_solutions(task.context, task.task, k=3)
        solutions_str = ""
        for k in range(len(recalled_solutions)):
            solutions_str += f"Solution {k}:\n" + self._recalled_solution_description(
                recalled_solutions[k]
            )
            if k < len(recalled_solutions) - 1:
                solutions_str += "\n\n"
        result = self._assess_solutions(task.task, task.context, solutions_str)
        task_with_solution_recall = _TaskWithSolutionRecall(
            task=task.task, context=task.context, solution_recall=result
        )
        return task_with_solution_recall

    def recalled_or_new(
        self,
        task_with_solution_recall: _TaskWithSolutionRecall,
    ) -> Literal["recalled", "new"]:
        """Determine if the further path in the graph should go through the recalled or new solution."""
        task = task_with_solution_recall
        result = self._model.invoke(
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

    def _recalled_solution_description(self, solution: _Solution) -> str:
        return (
            f"Task: {solution.task}\nContext: {solution.context}\nSolution:\n{solution.solution}\n"
        )

    def _assess_solutions(self, task: str, context: str, solutions: str) -> str:
        formatted_system_prompt = SOLUTION_RECALL_PROMPT.format(
            task=task, context=context, solutions=solutions
        )
        messages = [SystemMessage(content=formatted_system_prompt)]
        result = str(self._model.invoke(messages).content)
        return result
