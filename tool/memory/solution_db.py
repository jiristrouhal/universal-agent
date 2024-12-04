from typing import Any

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


class SolutionDB:

    COLLECTION_NAME = "tool_memories"

    def __init__(self, db_path: str = "") -> None:
        if bool(db_path):
            db_dir_path = str(db_path)
        else:
            db_dir_path = None
        self._db = Chroma(
            collection_name=self.COLLECTION_NAME,
            embedding_function=OpenAIEmbeddings(),  # type: ignore
            persist_directory=db_dir_path if db_dir_path else None,
        )

    def count(self):
        return self._db._collection.count()

    def delete(self, action_name: str) -> None:
        self._db.delete([action_name])

    def insert(self, description: str, content: Any) -> None:
        self._db.add_texts(
            texts=[description],
            ids=[content],
            metadatas=[{"content": content}],
        )

    def search(self, query: str, k: int = 1) -> list[tuple[str, float]]:
        vectordb_records = self._db.similarity_search_with_score(query, k=k)
        return [(doc.metadata["name"], score) for doc, score in vectordb_records]
