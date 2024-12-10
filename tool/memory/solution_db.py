from __future__ import annotations
import uuid
import json

import pydantic
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


class Solution(pydantic.BaseModel):
    """This class represents a solution to a specific task in a specific context. It is used to store the solution in the database
    with the data necessary for the solution modifications and verification, including tests, source links and the solution structure.
    """

    context: str
    task: str
    requirements: list[str]
    solution_structure: list[str]
    sources: dict[str, str]
    tests: list[Test]
    solution: str
    id: str = str(uuid.uuid4())

    @property
    def task_description(self) -> str:
        return f"Context: {self.context}\nTask: {self.task}"


class Test(pydantic.BaseModel):
    """This class represents a test case for a solution."""

    description: str
    implementation: str = ""
    critique_of_last_run: str = ""


class _SolutionDB:

    def __init__(self) -> None:
        self._db = Chroma(
            collection_name="solution",
            embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
            persist_directory="./data",
        )

    def add_solution(self, solution: Solution) -> Solution:
        id = self._db.add_texts(
            texts=[solution.task_description],
            metadatas=[{"json": solution.model_dump_json(indent=4)}],
        )[0]
        solution.id = id
        return solution

    def get_solutions(self, context: str, task: str, k: int = 3) -> list[Solution]:
        query = f"Context: {context}\nTask: {task}"
        return [
            Solution(**json.loads(d.metadata["json"]))
            for d in self._db.similarity_search(query, k=k)
        ]


database = _SolutionDB()
