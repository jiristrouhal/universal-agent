from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from tool.models import Solution as Solution, State as _State
from tool.memory.solution_db import get_solution_database as _get_database


_PROPOSE_SOLUTION_PROMPT = """
You are an experienced problem solver, that helps me to design solution to a task. I will provide you with a following information:

Context: ...
Task: ...
Tests: ...
Solution structure: ...
Information resources: ...

Please, follow these guidelines:
1) The solution must be relevant and testable in the context of the task.
2) Use only the provided Resources for building the solution. Do not make any assumptions or ad-hoc information retrievals.
3) All points from the Solution structure must be addressed in the solution.
4) The solution must pass all the Tests.
5) Write only the solution, helper functions and imports.
6) DO NOT write tests into the solution.
7) Do not write any additional information or code.
8) When writing function, provide type hints if applicable.
9) When the task is a simple question or request, provide a simple answer or solution.

You should respond to me only with the solution. Do not write anything else.
"""


class Proposer:

    def __init__(self, db_dir_path: str, openai_model: str = "gpt-4o-mini") -> None:
        self._db = _get_database(db_dir_path)
        self._model = ChatOpenAI(model=openai_model)

    def propose_solution(self, solution: Solution) -> Solution:
        """Propose a solution to the task and store it in the solution database."""
        query = (
            f"Context: {solution.context}\n"
            f"Task: {solution.task}\n"
            f"Tests: {solution.tests}\n"
            f"Solution structure: {solution.solution_structure}"
            f"Information resources: {solution.resources}"
        )
        messages = [SystemMessage(content=_PROPOSE_SOLUTION_PROMPT), HumanMessage(content=query)]
        solution_content = str(self._model.invoke(messages).content)
        solution.solution = solution_content
        self._db.add_solution(solution)
        return solution

    def print_solution(self, solution: Solution) -> _State:
        return _State(messages=[AIMessage(content=solution.solution)])
