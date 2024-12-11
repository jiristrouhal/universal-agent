from __future__ import annotations
import json

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from tool.models import Resource as _Resource, ResourceForm as _ResourceForm


class _ResourceDB:

    def __init__(self) -> None:
        self._db = {
            "code": self._create_db("code_db"),
            "text": self._create_db("text_db"),
        }

    def add(self, resource: _Resource) -> _Resource:
        retrieval_query = f"Context: {resource.context}\nTask: {resource.query}"
        id_ = self._db[resource.form].add_texts(
            texts=[retrieval_query],
            metadatas=[{"json": resource.model_dump_json(indent=4)}],
        )[0]
        resource.id = id_
        return resource

    def get(self, form: _ResourceForm, context: str, task: str, k: int = 3) -> list[_Resource]:
        query = f"Context: {context}\nTask: {task}"
        return [
            _Resource(**json.loads(d.metadata["json"]))
            for d in self._db[form].similarity_search(query, k=k)
        ]

    @staticmethod
    def _create_db(collection_name: str) -> Chroma:
        return Chroma(
            collection_name=collection_name,
            embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
            persist_directory="./data",
        )


database = _ResourceDB()
