from __future__ import annotations
from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, AnyMessage
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

    def recall(self, empty_solution: _Solution) -> _Solution:
        """Recall the solution from the memory and pick the most relevant. If none of the recalled solutions is relevant, return the empty solution."""
        solutions = self._db.get_solutions(empty_solution.context, empty_solution.task, k=3)
        picked = self._pick_solution(empty_solution.task, empty_solution.context, solutions)
        return picked

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

    def _pick_solution(self, task: str, context: str, solutions: list[_Solution]) -> _Solution:
        solutions_str = ""
        for i, solution in enumerate(solutions):
            solutions_str += f"{i}. {self._recalled_solution_description(solution)}\n"
        formatted_system_prompt = SOLUTION_RECALL_PROMPT.format(
            task=task, context=context, solutions=solutions_str
        )
        messages: list[AnyMessage] = [SystemMessage(content=formatted_system_prompt)]
        picked_solution = str(self._model.invoke(messages).content)
        messages.append(AIMessage(content=picked_solution))
        messages.append(
            HumanMessage(
                content="What is then the index of the best solution? Do not write anything else. Return 'None' if none of the solutions is relevant."
            )
        )
        result = str(self._model.invoke(messages))
        if result.isdigit():
            return solutions[int(result)]
        else:
            # Return the empty solution if none of the solutions is relevant.
            return _Solution(task=task, context=context)
