import unittest

from pprint import pprint
from IPython.display import Image

from tool.models import Solution, Test
from tool.validator.text import text_validator_builder


class Test_Text_Validator(unittest.TestCase):

    TEST_SOLUTION = Solution(
        context="",
        task="",
        requirements=[],
        solution_structure=[],
        resources={},
        solution="The highest point of the Czech Republic is the peak of Snezka, which is 1602 meters above sea level.",
        tests=[
            Test(description="The solution contains a sentence about the Czech Republic."),
            Test(description="The solution contains a sentence about Snezka."),
            Test(description="The solution contain any information on tourism in Czech Republic."),
        ],
    )

    def test_text_validator(self):
        graph = text_validator_builder.compile()
        with open("misc/graph.png", "wb") as f:
            f.write(Image(graph.get_graph().draw_mermaid_png()).data)
        result = graph.invoke(self.TEST_SOLUTION.model_dump())
        for test in result["tests"]:
            pprint(test.model_dump())
            print("\n")


if __name__ == "__main__":
    unittest.main()
