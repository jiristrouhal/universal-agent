import os
import shutil
import unittest
import uuid

from tool.models import Solution, Test
from tool.proposer import Proposer


class Test_Reiteration(unittest.TestCase):

    def setUp(self):
        self.test_db_path = os.path.join(os.path.dirname(__file__), f"./test_data_{uuid.uuid1()}")
        if os.path.exists(self.test_db_path):
            shutil.rmtree(self.test_db_path)
        self.proposer = Proposer(db_dir_path=self.test_db_path)

    def test_reiteration_with_failed_test_with_critique_returns_the_updated_solution(self):
        solution = Solution(
            context="I want to build library of unusual mathematical functions.",
            task="Give me a code returning the geometric average of a list of floats.",
            solution="""
def geometric_mean(lst: list[float]) -> float:
    prod = 1
    for num in lst:
        prod *= num
    return prod ** (1 / len(lst))
""",
            tests=[
                Test(
                    description="Function returns zero if list is empty.",
                    critique_of_last_run="The function does not handle case of empty list. Add a check for empty list and return 0.",
                ),
            ],
        )
        new_solution = self.proposer.propose_solution(solution)
        print(new_solution.solution)

    def tearDown(self):
        if os.path.exists(self.test_db_path):
            shutil.rmtree(self.test_db_path, ignore_errors=True)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
