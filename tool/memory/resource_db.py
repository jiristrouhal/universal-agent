from __future__ import annotations
import json

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from tool.models import Resource as _Resource, ResourceForm as _ResourceForm


class ResourceDB:

    def __init__(self, persist_directory: str) -> None:
        self._db = {
            "code": self._create_db("code_db", persist_directory),
            "text": self._create_db("text_db", persist_directory),
        }

    def add(self, resource: _Resource) -> _Resource:
        retrieval_query = f"Context: {resource.context}\nTask: {resource.query}"
        id_ = self._db[resource.form].add_texts(
            texts=[retrieval_query],
            metadatas=[{"json": resource.model_dump_json(indent=4)}],
        )[0]
        resource.id = id_
        return resource

    def get(self, form: _ResourceForm, context: str, query: str, k: int = 3) -> list[_Resource]:
        """Retrieve most relevant resource from memory.

        The `form` argument specifies the type of resource to retrieve.
        The `context` argument specifies the context of the resource.
        The `task` argument specifies the task of the resource.
        The `k` argument specifies the number of resources
        """
        query = f"Context: {context}\nTask: {query}"
        return [
            _Resource(**json.loads(d.metadata["json"]))
            for d in self._db[form].similarity_search(query, k=k)
        ]

    @staticmethod
    def _create_db(collection_name: str, persist_directory: str) -> Chroma:
        return Chroma(
            collection_name=collection_name,
            embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
            persist_directory=persist_directory,
        )


database = ResourceDB("./data")
