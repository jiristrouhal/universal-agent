import unittest
import os
import shutil
from uuid import uuid4

from tool.recaller import Recaller
from tool.models import Solution


TEST_DB_PATH = os.path.dirname(__file__) + f"/test_data{uuid4()}"


class Test_Recaller(unittest.TestCase):

    def setUp(self) -> None:
        if os.path.exists(TEST_DB_PATH):
            shutil.rmtree(TEST_DB_PATH, ignore_errors=True)
        self.recaller = Recaller(TEST_DB_PATH)

    def test_recall(self):
        empty_solution = Solution(
            task="Give me a code returning the geometric average of a list of floats.",
            context="I want to build library of unusual mathematical functions.",
        )
        recalled_solution = self.recaller.recall(empty_solution)
        self.assertEqual(recalled_solution, empty_solution)

    def tearDown(self):
        if os.path.exists(TEST_DB_PATH):
            shutil.rmtree(TEST_DB_PATH, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
