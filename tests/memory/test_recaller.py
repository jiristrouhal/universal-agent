import unittest
import os
import shutil

from tool.recaller import Recaller
from tool.models import TaskWithSolutionRecall


TEST_DB_PATH = os.path.dirname(__file__) + "/test_data"


class Test_Recaller(unittest.TestCase):

    def setUp(self) -> None:
        if os.path.exists(TEST_DB_PATH):
            shutil.rmtree(TEST_DB_PATH, ignore_errors=True)
        self.recaller = Recaller(TEST_DB_PATH)

    def test_recall(self):
        task_without_solution = TaskWithSolutionRecall(
            task="Give me a code returning the geometric average of a list of floats.",
            context="I want to build library of unusual mathematical functions.",
            solution_recall="",
        )
        recalled_solution = self.recaller.recall(task_without_solution)
        print(recalled_solution.solution_recall)

    def tearDown(self):
        if os.path.exists(TEST_DB_PATH):
            shutil.rmtree(TEST_DB_PATH, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
