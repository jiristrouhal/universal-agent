from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from tool.logs import get_logger
from tool.models import Solution as _Solution, State as _State
from tool.memory.solution_db import get_solution_database as _get_database


logger = get_logger()


_PROPOSE_SOLUTION_PROMPT = """
You are an experienced problem solver, that helps me to design solution to a task. I will provide you with a following information:

Context: ...
Task: ...
Solution structure: ...
Information resources: ...
Tests: ...
Previous solution: ...

Please, follow these guidelines:
- The solution must be relevant and testable in the context of the task.
- Use the provided Resources for building the solution. Do not make any assumptions or ad-hoc information retrievals.
- All points from the Solution structure must be addressed in the solution.
- The solution must pass all the Tests. Address all the critique from the last run of every test, that failed.
- Write only the solution. DO NOT write tests into the solution. Do not write anything else.
{form_specific_guidelines}
"""


CODE_GUIDELINES = """
- Add type hints to the function arguments and return value, for example
    def my_function(arg1: int, arg2: str) -> float:
       ...
- First write any helper functions if needed. Then write the main function solving the task.
- Do not write any tests or examples.
"""


TEXT_GUIDELINES = """
- Write the solution in a clear and concise manner.
"""


class Compiler:

    def __init__(self, db_dir_path: str, openai_model: str = "gpt-4o-mini") -> None:
        self._db = _get_database(db_dir_path)
        self._model = ChatOpenAI(model=openai_model)

    def compile(self, solution: _Solution) -> _Solution:
        """Propose a solution to the task and store it in the solution database."""
        query = (
            f"Context: {solution.context}\n"
            f"Task: {solution.task}\n"
            f"Solution structure: {solution.structure}"
            f"Information resources: {solution.resources}"
            f"Tests: {solution.tests}\n"
            f"Previous solution: {solution.solution}"
        )
        form_specific_guidelines = (
            CODE_GUIDELINES if solution.structure == "code" else TEXT_GUIDELINES
        )
        messages = [
            SystemMessage(
                content=_PROPOSE_SOLUTION_PROMPT.format(
                    form_specific_guidelines=form_specific_guidelines
                )
            ),
            HumanMessage(content=query),
        ]
        solution.proposal_tries += 1
        logger.debug(f"Compiling solution for task: {solution.task}")
        solution_content = str(self._model.invoke(messages).content)
        solution.solution = solution_content
        self._db.add_solution(solution)
        logger.debug(
            f"Compiled solution (attempt no. {solution.proposal_tries}): {solution_content}"
        )
        return solution

    def print_solution(self, solution: _Solution) -> _State:
        return _State(messages=[AIMessage(content=solution.solution)])
