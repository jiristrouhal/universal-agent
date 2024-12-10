from __future__ import annotations
import json

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from tool.models import Solution as _Solution


class _SolutionDB:

    def __init__(self) -> None:
        self._db = Chroma(
            collection_name="solution",
            embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
            persist_directory="./data",
        )

    def add_solution(self, solution: _Solution) -> _Solution:
        id_ = self._db.add_texts(
            texts=[solution.task_description],
            metadatas=[{"json": solution.model_dump_json(indent=4)}],
        )[0]
        solution.id = id_
        return solution

    def get_solutions(self, context: str, task: str, k: int = 3) -> list[_Solution]:
        query = f"Context: {context}\nTask: {task}"
        return [
            _Solution(**json.loads(d.metadata["json"]))
            for d in self._db.similarity_search(query, k=k)
        ]


database = _SolutionDB()
