import unittest

from tool.recaller import recall
from tool.models import TaskWithSolutionRecall


class Test_Recaller(unittest.TestCase):

    def test_recall(self):
        task_without_solution = TaskWithSolutionRecall(
            task="Give me a code returning the geometric average of a list of floats.",
            context="I want to build library of unusual mathematical functions.",
            solution_recall="",
        )
        recalled_solution = recall(task_without_solution)
        print(recalled_solution)


if __name__ == "__main__":
    unittest.main()
