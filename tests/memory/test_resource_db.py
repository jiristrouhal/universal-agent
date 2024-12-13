import os
import shutil
import unittest
from uuid import uuid4

from tool.memory.resource_db import ResourceDB
from tool.models import Resource


TEST_DB_PATH = os.path.dirname(__file__) + f"/test_data{uuid4()}"


class Test_ResourceDB(unittest.TestCase):

    def setUp(self):
        if os.path.isdir(TEST_DB_PATH):
            shutil.rmtree(TEST_DB_PATH, ignore_errors=True)
        self.db = ResourceDB(TEST_DB_PATH)

    def test_resource_db(self):
        self.db.add(
            Resource(
                form="text",
                context="Personal questions about work life",
                content="I work in the city of Brno.",
                request="In which city do I work?",
            )
        )
        results = self.db.get("text", "Personal questions about work life", "Where do I work?")
        assert "Brno" in results[0].content, "The result must contain the city of Brno."

    def tearDown(self):
        if os.path.isdir(TEST_DB_PATH):
            shutil.rmtree(TEST_DB_PATH, ignore_errors=True)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
