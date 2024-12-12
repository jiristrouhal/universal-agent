import unittest

from pprint import pprint
from IPython.display import Image

from tool.models import Solution, Test
from tool.validator.validator_graph import validator_builder


class Test_Validator_Graph(unittest.TestCase):

    TEST_SOLUTION_TEXT = Solution(
        context="",
        task="",
        requirements=[],
        solution_structure=[],
        resources={},
        form="text",
        solution="The highest point of the Czech Republic is the peak of Snezka, which is 1602 meters above sea level.",
        tests=[
            Test(description="The solution contains a sentence about the Czech Republic."),
            Test(description="The solution contains a sentence about Snezka."),
            Test(description="Does the solution contain anything about weather in Czech Republic?"),
        ],
    )

    TEST_SOLUTION_CODE = Solution(
        context="",
        task="",
        requirements=[],
        solution_structure=[],
        resources={},
        form="code",
        solution="def add_one(x: float) -> float:\n    return x + 1",
        tests=[
            Test(description="The result for input 5 should be 6."),
            Test(description="The result for input 10 should be 11."),
            Test(description="The result for input -15 should be -14."),
        ],
    )

    def test_text_validator(self):
        graph = validator_builder.compile()
        with open("misc/validator_graph.png", "wb") as f:
            f.write(Image(graph.get_graph(xray=True).draw_mermaid_png()).data)
        result = graph.invoke(self.TEST_SOLUTION_CODE.model_dump())
        pprint(result)

    def test_code_validator(self):
        graph = validator_builder.compile()
        with open("misc/validator_graph.png", "wb") as f:
            f.write(Image(graph.get_graph(xray=True).draw_mermaid_png()).data)
        result = graph.invoke(self.TEST_SOLUTION_TEXT.model_dump())
        pprint(result)


if __name__ == "__main__":
    unittest.main()
