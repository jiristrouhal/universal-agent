import unittest
from pprint import pprint

from tool.models import Solution, Test
from tool.validator.text import get_text_validator_builder


class Test_Text_Validator(unittest.TestCase):

    TEST_SOLUTION = Solution(
        context="",
        task="",
        requirements=[],
        structure=[],
        resources={},
        solution="The highest point of the Czech Republic is the peak of Snezka, which is 1602 meters above sea level.",
        tests=[
            Test(description="The solution contains a sentence about the Czech Republic."),
            Test(description="The solution contains a sentence about Snezka."),
            Test(description="The solution contain any information on tourism in Czech Republic."),
        ],
    )

    def test_text_validator(self):
        graph = get_text_validator_builder().compile()
        result = graph.invoke(self.TEST_SOLUTION.model_dump())
        for test in result["tests"]:
            pprint(test.model_dump())
            print("\n")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
