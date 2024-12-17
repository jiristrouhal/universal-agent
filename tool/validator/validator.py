from __future__ import annotations

import dotenv
from langchain_openai import ChatOpenAI

from tool.models import Solution as _Solution
from tool.validator.validator_graph import get_validator_builder as _get_validator_builder


dotenv.load_dotenv()


class Validator:

    def __init__(self, openai_model: str = "gpt-4o-mini") -> None:
        self._model = ChatOpenAI(model=openai_model)
        self._compile_graph()

    def _compile_graph(self) -> None:
        builder = _get_validator_builder()
        self._graph = builder.compile()

    def review(self, solution: _Solution) -> None:
        result = self._graph.invoke(solution.model_dump(), {"recursion_limit": 50})
        for key, value in result.items():
            setattr(solution, key, value)
