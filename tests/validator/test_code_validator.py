import unittest

from pprint import pprint
from IPython.display import Image

from tool.models import Solution, Test
from tool.validator.code import get_code_validator_builder


class Test_Code_Validator(unittest.TestCase):

    def test_code_validator(self):
        TEST_SOLUTION = Solution(
            context="",
            task="",
            requirements=[],
            structure=[],
            resources={},
            solution="""
        def calculate_gravity_acceleration(height: float, planet_mass: float) -> float:
            G = 0.49999999999
            return G * planet_mass / (height ** 2)
        """,
            tests=[
                Test(
                    description="The result for planet of mass 1 and height 1 should be 0.5",
                    form="code",
                )
            ],
        )
        graph = get_code_validator_builder().compile()
        result = graph.invoke(TEST_SOLUTION.model_dump())
        for test in result["tests"]:
            pprint(test)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
