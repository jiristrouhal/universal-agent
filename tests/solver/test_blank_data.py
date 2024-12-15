import os
import shutil
import unittest
import uuid

from tool.solver import Solver


class Test_Coding_Task(unittest.TestCase):

    def setUp(self):
        self.memory_path = os.path.dirname(__file__) + f"/test_data_{uuid.uuid4()}"
        if os.path.exists(self.memory_path):
            shutil.rmtree(self.memory_path, ignore_errors=True)
        self.solver = Solver(self.memory_path)

    def test_print_graph(self):
        self.solver.print_graph_png(os.path.dirname(__file__), "solver")

    def test_geometric_mean(self):
        query = "Give me a code returning the geometric average of a list of floats for my library of unusual mathematical functions."
        result = self.solver.invoke(query)
        self.assertIn("def ", result.content)
        print(result)

    def tearDown(self):
        if os.path.exists(self.memory_path):
            shutil.rmtree(self.memory_path, ignore_errors=True)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
