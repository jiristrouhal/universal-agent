from __future__ import annotations
from typing import Literal
import json

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, AnyMessage
from langchain_openai import ChatOpenAI

from tool.models import Solution as _Solution
from tool.memory.solution_db import (
    get_solution_database as _get_database,
    SolutionDB as _SolutionDB,
)


SOLUTION_RECALL_PROMPT = """
You are a helpful assistant, that analyzes the user's request and provides the most relevant solutions from memory.

Look on the given task asked in given context and assess the best solution from the list of solutions to similar problems from the past.
It is possible none of them are relevant, in that case, you respond with a message that there is no relevant solution.

Your answer must be in the following format:
Reasoning: Here you think step by step about Recalled solutions with respect to the task and context.
Best solution: Here you choose the best solution. Do not write anything else. It is possible that none of the solutions is relevant.

The input information for you is:

Task:\n{task}
Context:\n{context}
Solution requirements: {requirements}
Recalled solutions:\n{solutions}

You can start now.
"""


SOLUTION_OK_PROMPT = """
You are a helpful assistant that determines if the solution provided is acceptable.

I will provide to you the following information:

Task: ...
Context: ...
Solution recall: ...

You will respond with 'True' if the solution is nonempty and evaluated as valid, 'False' otherwise. Do not write anything else.
"""


USABLE_SOLUTION_PROMPT = """
You are a helpful assistant that determines if some of the solutions can be used directly given list of requirements and the task.

I will provide to you the following information:

Task: ...
Context: ...
Solution requirements: ...

Recalled solutions:
...

You must pick the first Recalled solution, whose requirements contain all the Solution requirements.
You will respond with an index of that solution. If none of the solutions is relevant, return 'None'.
"""


class Recaller:

    def __init__(self, db_dir_path: str, openai_model: str = "gpt-4o-mini") -> None:
        self._db = _get_database(db_dir_path)
        self._model = ChatOpenAI(model=openai_model)

    @property
    def solution_db(self) -> _SolutionDB:
        return self._db

    def recall(self, empty_solution: _Solution) -> _Solution:
        """Recall the solution from the memory and pick the most relevant. If none of the recalled solutions is relevant, return an empty solution."""
        assert empty_solution.solution == "", "The solution must be empty."
        assert empty_solution.task != "", "The task must be provided."
        assert empty_solution.context != "", "The context must be provided."
        assert len(empty_solution.requirements) > 0, "The requirements must be provided."
        solutions: list[_Solution] = self._db.get_solutions(
            empty_solution.task, empty_solution.context, empty_solution.requirements, k=3
        )
        directly_usable_solution_index = self._pick_directly_usable_solution(
            empty_solution, solutions
        )
        # check solution form
        if directly_usable_solution_index is not None:
            solution = solutions[directly_usable_solution_index]
            if solution.form == empty_solution.form:
                solution.task = empty_solution.task
                solution.context = empty_solution.context
                solution.requirements = empty_solution.requirements
                return solution

        picked = self._pick_solutions(empty_solution, solutions)
        picked_solutions_contents = ",\n".join(s.solution for s in picked)
        empty_solution.similar_solutions = picked_solutions_contents
        return empty_solution

    def recalled_or_new(
        self,
        solution_after_recall: _Solution,
    ) -> Literal["recalled", "new"]:
        """Determine if the further path in the graph should go through the recalled or new solution."""
        if solution_after_recall.solution:
            return "recalled"
        return "new"

    def _pick_directly_usable_solution(
        self, empty_solution: _Solution, solutions: list[_Solution]
    ) -> int | None:

        recalled_solutions_str = "\n"
        for k in range(len(solutions)):
            recalled_solutions_str += f"{k:2}. Solution:\n" + self._recalled_solution_description(
                solutions[k]
            )
        query = (
            f"Task: {empty_solution.task}\n"
            f"Context: {empty_solution.context}\n"
            f"Solution requirements: {', '.join(empty_solution.requirements)}\n"
            f"Recalled solutions: {recalled_solutions_str}"
        )
        messages = [SystemMessage(content=USABLE_SOLUTION_PROMPT), HumanMessage(content=query)]
        response = str(self._model.invoke(messages).content)
        if response.isdigit():
            return int(response)
        return None

    def _recalled_solution_description(self, solution: _Solution) -> str:
        return f"Task: {solution.task}\nContext: {solution.context}\nRequirements: {', '.join(solution.requirements)}"

    def _pick_solutions(
        self, empty_solution: _Solution, solutions: list[_Solution]
    ) -> list[_Solution]:
        solutions_str = ""
        for i, solution in enumerate(solutions):
            solutions_str += f"Solution {i}:\n{self._recalled_solution_description(solution)}\n"

        requirements_str = ", ".join(empty_solution.requirements)
        formatted_system_prompt = SOLUTION_RECALL_PROMPT.format(
            task=empty_solution.task,
            context=empty_solution.context,
            requirements=requirements_str,
            solutions=solutions_str,
        )
        messages: list[AnyMessage] = [SystemMessage(content=formatted_system_prompt)]
        picked_solution = str(self._model.invoke(messages).content)
        messages.append(AIMessage(content=picked_solution))
        messages.append(
            HumanMessage(
                content="What are the indices of the useful solutions? Return a list of integers separated by commas. If none of the solutions is relevant, return an empty list."
            )
        )
        result = str(self._model.invoke(messages).content)
        indices = [int(i) for i in json.loads(result) if str(i).isdigit()]
        return [solutions[int(i)] for i in indices if 0 <= i < len(solutions)]
