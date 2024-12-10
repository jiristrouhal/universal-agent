from pprint import pprint
from IPython.display import Image

from tool.models import Solution, Test
from tool.validator.validator_graph import validator_builder


TEST_SOLUTION = Solution(
    context="",
    task="",
    requirements=[],
    solution_structure=[],
    sources={},
    solution="The highest point of the Czech Republic is the peak of Snezka, which is 1602 meters above sea level.",
    tests=[
        Test(description="The solution contains a sentence about the Czech Republic.", form="text"),
        Test(description="The solution contains a sentence about Snezka.", form="text"),
        Test(
            description="Does the solution contain anything related to an  weather extremes in Czech Republic?",
            form="text",
        ),
    ],
)


def test_text_validator():
    graph = validator_builder.compile()
    with open("misc/graph.png", "wb") as f:
        f.write(Image(graph.get_graph().draw_mermaid_png()).data)
    result = graph.invoke(TEST_SOLUTION.model_dump())
    pprint(result)


if __name__ == "__main__":
    test_text_validator()
