from pprint import pprint
from IPython.display import Image

from tool.models import Solution, Test
from tool.validator.code import code_validator_builder


TEST_SOLUTION = Solution(
    context="",
    task="",
    requirements=[],
    solution_structure=[],
    sources={},
    solution="""
def calculate_gravity_acceleration(height: float, planet_mass: float) -> float:
    G = 0.49999999999
    return G * planet_mass / (height ** 2) + 0.01
""",
    tests=[
        Test(description="The result for planet of mass 1 and height 1 should be 0.5", form="code")
    ],
)


def test_code_validator():
    graph = code_validator_builder.compile()
    with open("misc/code_validator_graph.png", "wb") as f:
        f.write(Image(graph.get_graph().draw_mermaid_png()).data)
    result = graph.invoke(TEST_SOLUTION.model_dump())
    for test in result["tests"]:
        pprint(test)


if __name__ == "__main__":
    test_code_validator()
