import os
import shutil
import unittest

from tool.memory.resource_db import ResourceDB
from tool.models import Resource


test_db = ResourceDB(os.path.dirname(__file__) + "/test_data")


class Test_ResourceDB(unittest.TestCase):

    def test_resource_db(self):
        test_db.add(
            Resource(
                form="text",
                context="Personal questions about work life",
                content="I work in the city of Brno.",
                request="In which city do I work?",
            )
        )
        results = test_db.get("text", "Personal questions about work life", "Where do I work?")
        assert "Brno" in results[0].content, "The result must contain the city of Brno."

        if os.path.isdir("./tests/memory/test_data"):
            shutil.rmtree("./tests/memory/test_data", ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
