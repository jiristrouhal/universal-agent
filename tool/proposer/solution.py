from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from tool.models import (
    TaskWithResources as _TaskWithResources,
    State as _State,
    Solution as _Solution,
)
from tool.memory.solution_db import get_solution_database as _get_database


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


class Proposer:

    def __init__(self, db_dir_path: str, openai_model: str = "gpt-4o-mini") -> None:
        self._db = _get_database(db_dir_path)
        self._model = ChatOpenAI(model=openai_model)

    def propose_solution(self, draft: _TaskWithResources) -> _Solution:
        """Propose a solution to the task and store it in the solution database."""
        query = (
            f"Context: {draft.context}\n"
            f"Task: {draft.task}\n"
            f"Tests: {draft.tests}\n"
            f"Solution structure: {draft.solution_structure}"
        )
        messages = [SystemMessage(content=_PROPOSE_SOLUTION_PROMPT), HumanMessage(content=query)]
        solution = str(self._model.invoke(messages).content)
        solution_obj = _Solution(
            task=draft.task,
            context=draft.context,
            requirements=draft.requirements,
            resources=draft.resources,
            solution_structure=draft.solution_structure,
            tests=draft.tests,
            solution=solution,
        )
        self._db.add_solution(solution_obj)
        return solution_obj

    def print_solution(self, solution: _Solution) -> _State:
        return _State(messages=[AIMessage(content=solution.solution)])
