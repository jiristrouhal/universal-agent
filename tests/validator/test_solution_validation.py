import unittest

from tool.models import Solution, Test
from tool.validator import Validator


class Test_Validator(unittest.TestCase):

    def test_text_task_validator(self):
        solution = Solution(
            context="",
            task="",
            requirements=[],
            solution_structure=[],
            resources={},
            solution="The highest point of the Czech Republic is the peak of Snezka, which is 1602 meters above sea level.",
            tests=[
                Test(description="The solution contains a sentence about Snezka."),
                Test(description="The solution contains an incorrect sentence about Snezka."),
                Test(
                    description="The solution contains an information on tourism in Czech Republic."
                ),
            ],
        )
        validator = Validator()
        validator.review(solution)
        for t in solution.tests:
            print(t.model_dump_json(indent=4))
            print("\n")

    def test_task_for_returning_new_name_not_contained_in_given_list(self):
        solution = Solution(
            context="",
            task="",
            requirements=[],
            solution_structure=[],
            resources={},
            solution="Hugo, Peter, Alice.",
            tests=[
                Test(description="The solution does not contain Bob."),
                Test(description="The solution does contain Alice."),
            ],
        )
        validator = Validator()
        validator.review(solution)
        for t in solution.tests:
            print(t.model_dump_json(indent=4))
            print("\n")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
