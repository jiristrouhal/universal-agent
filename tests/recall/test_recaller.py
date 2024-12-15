import unittest
import os
import shutil
from uuid import uuid4

from tool.memory.recaller import Recaller
from tool.models import Solution


class Test_Recaller(unittest.TestCase):

    def setUp(self) -> None:
        self.db_path = os.path.dirname(__file__) + f"/test_data{uuid4()}"
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path, ignore_errors=True)
        self.recaller = Recaller(self.db_path)

    def test_empty_solution_db_yields_empty_solution(self):
        empty_solution = Solution(
            task="Give me a code returning the geometric average of a list of floats.",
            context="I want to build library of unusual mathematical functions.",
            requirements=[
                "The function takes only positive floats.",
                "The function returns zero for a list without elements.",
            ],
        )
        recalled_solution = self.recaller.recall(empty_solution)
        self.assertEqual(recalled_solution, empty_solution)

    def test_single_relevant_solution_is_recalled(self):
        self.recaller._db.add_solution(
            Solution(
                task="Give me a code returning the geometric average of a list of floats.",
                context="I want to build library of unusual mathematical functions.",
                solution="def geometric_average(lst):\n    return sum(lst) ** (1 / len(lst))",
                requirements=[
                    "The function takes only positive floats.",
                ],
            )
        )
        self.recaller._db.add_solution(
            Solution(
                task="Give me a code returning the geometric average of a list of floats.",
                context="I want to build library of unusual mathematical functions.",
                solution="def geometric_average(lst):\n    return sum(lst) ** (1 / len(lst))",
                requirements=[
                    "The function takes only positive floats.",
                    "The function returns zero for an empty list.",
                    "The function returns zero for an list of zeros.",
                ],
            ),
        )
        empty_solution = Solution(
            task="Give me a code returning the geometric average of a list of floats.",
            context="I want to build library of unusual mathematical functions.",
            requirements=[
                "The function takes only positive floats.",
                "The function returns zero for a list without elements.",
            ],
        )
        recalled = self.recaller.recall(empty_solution)
        self.assertEqual(
            recalled.context, "I want to build library of unusual mathematical functions."
        )
        self.assertEqual(
            recalled.task, "Give me a code returning the geometric average of a list of floats."
        )
        self.assertEqual(len(recalled.requirements), 2)
        self.assertEqual(
            recalled.requirements,
            [
                "The function takes only positive floats.",
                "The function returns zero for a list without elements.",
            ],
        )

    def tearDown(self):
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path, ignore_errors=True)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
