from __future__ import annotations
import json
import os

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from tool.models import Solution as _Solution


class SolutionDB:

    def __init__(self, persist_directory: str) -> None:
        self._db = Chroma(
            collection_name="solution",
            embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
            persist_directory=persist_directory,
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


def get_solution_database(path: str) -> SolutionDB:
    """Create a SolutionDB instance, providing access to a Chroma vector database (https://www.trychroma.com/).
    The database is created in the directory specified by the `path` argument.

    If the directory does not exist, it is created including its parent directories.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return SolutionDB(path)
