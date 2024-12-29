import os
import shutil
import unittest
import uuid

from tool import Assistant
from tool.models import Solution


class Test_Task_Solving_And_Validation(unittest.TestCase):

    def setUp(self):
        self.memory_path = os.path.dirname(__file__) + f"/test_data_{uuid.uuid4()}"
        if os.path.exists(self.memory_path):
            shutil.rmtree(self.memory_path, ignore_errors=True)
        self.assistant = Assistant(self.memory_path)

    def test_print_graph(self):
        self.assistant.print_graph_png(os.path.dirname(__file__), "tool")

    def test_a_summary_on_a_given_topic(self):
        query = "Write to me about 1000 words on the topic of a discovery of Amatérská jeskyně in Czech Republic, including people who participated on the discovery."
        result = self.assistant.invoke(query)
        print(result)

    def test_geometric_mean_with_blank_memory(self):
        query = "Give me a code returning the geometric average of a list of floats for my library of unusual mathematical functions."
        result = self.assistant.invoke(query)
        print(result)

    def test_graph_with_already_existing_solution_for_geometric_mean(self):
        solution = Solution(
            task="Give me a code returning the geometric average of a list of floats for my library of unusual mathematical functions.",
            context="",
            requirements=[
                "the solution must calculate the geometric average accurately",
                "the input must be a list of floats",
                "the solution must handle lists containing zero values appropriately",
                "the solution must not return an error for negative values, but should define the behavior clearly",
                "the solution must be efficient in terms of time complexity",
                "the solution must include error handling for invalid inputs",
                "the solution must return a float as the output",
                "the solution must document the method used for calculation",
                "the solution must be based on collected Resources",
            ],
            content='```python\nimport math\n\ndef geometric_average(numbers):\n    """\n    Calculate the geometric average of a list of floats.\n    \n    The geometric average is defined as:\n    G = (x₁ * x₂ * ... * xₙ)^(1/n)\n    Where G is the geometric average, x₁, x₂, ..., xₙ are the values in the list, \n    and n is the number of values.\n\n    Special cases:\n    - If the list contains zero values, the geometric average will be zero.\n    - The geometric average is not defined for negative values. If negative values \n      are present, an error will be raised.\n    \n    :param numbers: List of floats\n    :return: Geometric average as a float\n    :raises ValueError: If input is not a list of floats or if negative values are present.\n    """\n    \n    # Input validation\n    if not isinstance(numbers, list) or not all(isinstance(i, float) for i in numbers):\n        raise ValueError("Input must be a list of floats.")\n    \n    # Check for negative values\n    if any(num < 0 for num in numbers):\n        raise ValueError("Geometric average is not defined for negative values.")\n    \n    # Handle zero values\n    if any(num == 0 for num in numbers):\n        return 0.0\n    \n    # Calculate the geometric average\n    product = 1\n    for number in numbers:\n        product *= number\n    \n    return product ** (1 / len(numbers))\n```\n',
        )
        self.assistant.solution_db.add_solution(solution)
        query = "Give me a code, that returns either arithmetic or geometric average of a list of floats, based on a literal argument 'arithmetic' or 'geometric'."
        result = self.assistant.invoke(query)
        print(result)

    def tearDown(self):
        if os.path.exists(self.memory_path):
            shutil.rmtree(self.memory_path, ignore_errors=True)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
