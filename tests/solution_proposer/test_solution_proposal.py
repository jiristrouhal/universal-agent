import unittest

from tool.proposer import Proposer
from tool.models import Solution, Test


class Test_Solution_Proposal(unittest.TestCase):

    def setUp(self):
        self.proposer = Proposer(db_dir_path="./data/solutions")

    def test_solution_proposal(self):
        draft = Solution(
            task="Give me a code returning the geometric average of a list of floats.",
            context="I want to build library of unusual mathematical functions.",
            tests=[
                Test(description="assert geometric_average([1.0, 2.0, 3.0]) == 1.8171205928321397"),
                Test(
                    description="assert geometric_average([1.0, 2.0, 3.0, 4.0]) == 2.213363839400643"
                ),
                Test(description="assert geometric_average([0, 0, 0, 0]) == 0"),
                Test(description="assert geometric_average([]) == 0.0"),
            ],
            requirements=["The function must return zero if the list is empty."],
            solution_structure=[
                "Check if the list does not contain negative numbers.",
                "Check if the list is empty and return 0 if it does",
                "Calculate the product of the list elements",
                "Calculate the length of the list and store the value as n."
                "Return the nth root of the product, where n is the length of the list",
            ],
            resources={},
        )

        solution = self.proposer.propose_solution(draft)
        assert isinstance(solution, Solution), "The solution must be a Solution object."
        print(solution.solution)


if __name__ == "__main__":
    unittest.main()
