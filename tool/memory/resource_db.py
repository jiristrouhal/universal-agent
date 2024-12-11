from __future__ import annotations
import json
import os

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from tool.models import Resource as _Resource, ResourceForm as _ResourceForm


class ResourceDB:

    def __init__(self, persist_directory: str) -> None:
        self._db = {
            "code": self.create_db("code_db", persist_directory),
            "text": self.create_db("text_db", persist_directory),
        }

    def add(self, resource: _Resource) -> _Resource:
        retrieval_query = f"Context: {resource.context}\nTask: {resource.request}"
        id_ = self._db[resource.form].add_texts(
            texts=[retrieval_query],
            metadatas=[{"json": resource.model_dump_json(indent=4)}],
        )[0]
        resource.id = id_
        return resource

    def get(self, form: _ResourceForm, context: str, request: str, k: int = 3) -> list[_Resource]:
        """Retrieve most relevant resource from memory.

        The `form` argument specifies the type of resource to retrieve.
        The `context` argument specifies the context of the resource.
        The `task` argument specifies the task of the resource.
        The `k` argument specifies the number of resources
        """
        request = f"Context: {context}\nTask: {request}"
        return [
            _Resource(**json.loads(d.metadata["json"]))
            for d in self._db[form].similarity_search(request, k=k)
        ]

    @staticmethod
    def create_db(collection_name: str, persist_directory: str) -> Chroma:
        return Chroma(
            collection_name=collection_name,
            embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
            persist_directory=persist_directory,
        )


def new_custom_database(db_location: str = "") -> ResourceDB:
    return ResourceDB(db_location)


database = ResourceDB("./data")
