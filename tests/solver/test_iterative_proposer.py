import os
import shutil
import unittest
import uuid

from tool.models import Solution, Test
from tool.solver.solver import IterativeProposer
from tool.proposer.tests import get_tests


class Test_Coding_Task(unittest.TestCase):

    def setUp(self):
        self.memory_path = os.path.dirname(__file__) + f"/test_data_{uuid.uuid4()}"
        if os.path.exists(self.memory_path):
            shutil.rmtree(self.memory_path, ignore_errors=True)
        self.solver = IterativeProposer(self.memory_path)

    def test_print_graph(self):
        self.solver.print_graph_png(os.path.dirname(__file__), "iterative_proposer")

    def test_geometric_mean(self):
        solution = Solution(
            task="Give me a code returning the geometric average of a list of floats for my library of unusual mathematical functions.",
            context="I am a mathematician who is working on a library of unusual mathematical functions.",
            solution="def geometric_mean(lst):\n    return (reduce(lambda x, y: x * y, lst)) ** (1 / len(lst))",
            requirements=[
                "The function should return 0 for empty list",
                "The function should return 0 for list with 0",
                "The function should raise error for list with negative numbers",
            ],
        )
        solution = get_tests(solution)
        result = self.solver.invoke(solution)
        print(result)

    def tearDown(self):
        if os.path.exists(self.memory_path):
            shutil.rmtree(self.memory_path, ignore_errors=True)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
