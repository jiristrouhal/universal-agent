import unittest

from tool.models import Solution, Test
from tool.validator import Validator


class Test_Validator(unittest.TestCase):

    def test_text_task_validator(self):
        solution = Solution(
            context="",
            task="",
            requirements=[],
            structure=[],
            resources={},
            solution="The highest point of the Czech Republic is the peak of Snezka, which is 1602 meters above sea level.",
            tests=[
                Test(description="The solution contains a sentence about Snezka."),
                Test(description="The solution contains an incorrect sentence about Snezka."),
                Test(
                    description="The solution contains an information on tourism in Czech Republic."
                ),
                Test(
                    description="The solution must contain an information related to some popular touristic place in Czech Republic."
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
            structure=[],
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

    def test_coding_task(self):
        solution_dict = {
            "context": "Calculate gravitational acceleration above a planet's surface",
            "task": "Write a Python function that calculates gravitational acceleration",
            "requirements": [
                "The function must calculate gravitational acceleration using the formula g = G * (M / r^2), where G is the gravitational constant, M is the mass of the planet, and r is the distance from the center of the planet to the point where gravitational acceleration is being calculated.",
                "The function must accept parameters for the mass of the planet and the distance from the center of the planet.",
                "The function must return the gravitational acceleration as a float.",
                "The function must handle edge cases, such as when the distance from the center of the planet is less than or equal to zero, by raising a ValueError.",
                "The function must be well-documented with comments explaining the logic and the parameters used.",
            ],
            "structure": [
                "Part 1: Define the function with parameters for mass and distance",
                "Part 2: Check if the distance is less than or equal to zero and raise ValueError if true",
                "Part 3: Define the gravitational constant G",
                "Part 4: Calculate gravitational acceleration using the formula g = G * (M / r^2)",
                "Part 5: Return the calculated gravitational acceleration as a float",
                "Part 6: Add comments explaining the logic and parameters used in the function",
            ],
            "resources": {
                "I need to find the formula for gravitational acceleration. I expect to get it as a plain text description.": "The formula for gravitational acceleration \\( g \\) above a planet's surface can be derived from Newton's law of universal gravitation. It is given by the following equation:\n\n\\[\ng = \\frac{G \\cdot M}{r^2}\n\\]\n\nWhere:\n- \\( g \\) is the gravitational acceleration,\n- \\( G \\) is the gravitational constant (\\( 6.674 \\times 10^{-11} \\, \\text{m}^3 \\cdot \\text{kg}^{-1} \\cdot \\text{s}^{-2} \\)),\n- \\( M \\) is the mass of the planet,\n- \\( r \\) is the distance from the center of the planet to the point where gravitational acceleration is being calculated (which is the radius of the planet plus the height above the surface).\n\nThis formula indicates that gravitational acceleration decreases with the square of the distance from the center of the planet.",
                "I need the value of the gravitational constant G. I expect to get it as a plain text value.": "The value of the gravitational constant \\( G \\) is approximately \\( 6.6743 \\times 10^{-11} \\, \\text{N} \\cdot \\text{m}^2/\\text{kg}^2 \\).",
                "I need to understand how to handle exceptions in Python, specifically raising ValueError. I expect to get it as a plain text explanation.": 'In Python, exceptions are a way to handle errors that occur during the execution of a program. Specifically, a `ValueError` is raised when a function receives an argument of the right type but an inappropriate value. For instance, if a function expects a number within a specific range and receives a number outside that range, it can raise a `ValueError` to indicate that the input is not valid.\n\nHere’s how you can work with exceptions, including raising a `ValueError`:\n\n1. **Using `try` and `except` Blocks**: You can wrap the code that may cause an exception in a `try` block, and handle the exception in the corresponding `except` block.\n\n2. **Raising a `ValueError`**: If you detect that an input value does not meet the expected criteria, you can raise a `ValueError` using the `raise` statement. You can also provide a message that explains the reason for the exception.\n\nHere’s a simple example:\n\n```python\ndef calculate_gravity(mass, radius):\n    if mass <= 0:\n        raise ValueError("Mass must be greater than zero.")\n    if radius <= 0:\n        raise ValueError("Radius must be greater than zero.")\n    \n    G = 6.67430e-11  # Gravitational constant\n    return G * mass / (radius ** 2)\n\n# Example usage:\ntry:\n    print(calculate_gravity(5.972e24, 6371000))  # Earth mass and radius\n    print(calculate_gravity(-5.972e24, 6371000))  # This will raise a ValueError\nexcept ValueError as e:\n    print("Error:", e)\n```\n\nIn this example, if the `mass` or `radius` is less than or equal to zero, a `ValueError` is raised with an appropriate message. The `try` block is used to attempt the function call, and if a `ValueError` occurs, it is caught in the `except` block, allowing for graceful handling of the error.',
                "I need examples of Python functions and how to define them with parameters. I expect to get it as Python code snippets.": "Here are examples of Python functions that can be used to calculate gravitational acceleration above a planet's surface, along with explanations of how to define them with parameters:\n\n### Example 1: Simple Gravitational Acceleration Function\nThis function calculates gravitational acceleration using the formula \\( g = \\frac{G \\cdot M}{r^2} \\), where:\n- \\( g \\) is the gravitational acceleration,\n- \\( G \\) is the gravitational constant (approximately \\( 6.674 \\times 10^{-11} \\, \\text{m}^3/\\text{kg}\\cdot\\text{s}^2 \\)),\n- \\( M \\) is the mass of the planet,\n- \\( r \\) is the distance from the center of the planet to the object.\n\n```python\ndef gravitational_acceleration(M, r):\n    G = 6.674 * 10**-11  # gravitational constant in m^3 kg^-1 s^-2\n    g = G * M / r**2  # gravitational acceleration\n    return g\n```\n\n### Example 2: Gravitational Acceleration Above Planet's Surface\nThis function calculates the gravitational acceleration at a height above the planet's surface. It takes the mass of the planet, the radius of the planet, and the height above the surface as parameters.\n\n```python\ndef gravitational_acceleration_at_height(M, R, h):\n    G = 6.674 * 10**-11  # gravitational constant in m^3 kg^-1 s^-2\n    r = R + h  # total distance from the center of the planet\n    g = G * M / r**2  # gravitational acceleration at height h\n    return g\n```\n\n### Example 3: Gravitational Acceleration for Different Planets\nThis function allows you to calculate the gravitational acceleration for different planets by passing the mass and radius of the planet as parameters.\n\n```python\ndef gravitational_acceleration_planet(M, R):\n    G = 6.674 * 10**-11  # gravitational constant in m^3 kg^-1 s^-2\n    g = G * M / R**2  # gravitational acceleration at the surface\n    return g\n```\n\n### Usage Example\nHere's how you might call one of the functions to calculate gravitational acceleration:\n\n```python\nmass_earth = 5.972 * 10**24  # mass of Earth in kg\nradius_earth = 6371000  # radius of Earth in meters\nheight_above_surface = 1000  # height in meters\n\n# Calculate gravitational acceleration at a height of 1000 meters above Earth's surface\ng_at_height = gravitational_acceleration_at_height(mass_earth, radius_earth, height_above_surface)\nprint(f\"Gravitational acceleration at {height_above_surface} meters above Earth's surface: {g_at_height} m/s²\")\n```\n\nThese examples illustrate how to define Python functions with parameters to calculate gravitational acceleration in different scenarios.",
                "I need examples of how to write comments in Python code. I expect to get it as a plain text explanation.": "In Python, comments are used to explain the code and make it more readable. They are ignored by the interpreter and do not affect the execution of the program. Here are the different ways to write comments in Python:\n\n1. **Single-line comments**: You can create a single-line comment by placing a hash symbol `#` before the comment text. Everything after the `#` on that line will be treated as a comment.\n\n   Example:\n   ```python\n   # This is a single-line comment\n   gravitational_constant = 9.81  # Acceleration due to gravity on Earth\n   ```\n\n2. **Multi-line comments**: Although Python does not have a specific syntax for multi-line comments, you can use triple quotes (`'''` or `\"\"\"`) to create a multi-line string that is not assigned to any variable. This string will be ignored by the interpreter, effectively serving as a comment.\n\n   Example:\n   ```python\n   '''\n   This is a multi-line comment.\n   It can span multiple lines.\n   '''\n   gravitational_constant = 9.81\n   ```\n\n3. **Docstrings**: These are special types of comments that are used to describe functions, classes, and modules. They are written using triple quotes and can be accessed through the `__doc__` attribute. Docstrings are a great way to document what a function does.\n\n   Example:\n   ```python\n   def calculate_gravity(mass, radius):\n       \"\"\"\n       Calculate the gravitational acceleration at a given distance from the center of a planet.\n\n       Parameters:\n       mass (float): Mass of the planet in kilograms.\n       radius (float): Distance from the center of the planet in meters.\n\n       Returns:\n       float: Gravitational acceleration in m/s^2.\n       \"\"\"\n       gravitational_constant = 6.67430e-11  # Universal gravitational constant\n       return gravitational_constant * mass / (radius ** 2)\n   ```\n\nIn summary, you can use `#` for single-line comments, triple quotes for multi-line comments, and docstrings for documentation within functions or classes.",
            },
            "tests": [
                {
                    "description": "Test 1: Calculate gravitational acceleration for a planet with mass 5.972e24 kg (Earth) at a distance of 6.371e6 m (Earth's radius). Expected result: approximately 9.81 m/s^2.",
                    "implementation": "",
                    "form": "code",
                    "last_output": "",
                    "critique_of_last_run": "",
                    "result": "unknown",
                },
                {
                    "description": "Test 2: Calculate gravitational acceleration for a planet with mass 7.348e22 kg (Moon) at a distance of 1.737e6 m (Moon's radius). Expected result: approximately 1.62 m/s^2.",
                    "implementation": "",
                    "form": "code",
                    "last_output": "",
                    "critique_of_last_run": "",
                    "result": "unknown",
                },
                {
                    "description": "Test 3: Handle edge case where distance from the center of the planet is less than or equal to zero (e.g., distance = 0 m). Expected result: raise ValueError.",
                    "implementation": "",
                    "form": "code",
                    "last_output": "",
                    "critique_of_last_run": "",
                    "result": "unknown",
                },
                {
                    "description": "Test 4: Calculate gravitational acceleration for a planet with mass 1.989e30 kg (Sun) at a distance of 6.957e8 m (Sun's radius). Expected result: approximately 274 m/s^2.",
                    "implementation": "",
                    "form": "code",
                    "last_output": "",
                    "critique_of_last_run": "",
                    "result": "unknown",
                },
                {
                    "description": "Test 5: Check the return type of the function for valid inputs (e.g., mass = 5.972e24 kg, distance = 6.371e6 m). Expected result: return type is float.",
                    "implementation": "",
                    "form": "code",
                    "last_output": "",
                    "critique_of_last_run": "",
                    "result": "unknown",
                },
            ],
            "form": "code",
            "solution": '```python\ndef calculate_gravitational_acceleration(mass, distance):\n    """\n    Calculate gravitational acceleration above a planet\'s surface.\n\n    Parameters:\n    mass (float): Mass of the planet in kilograms.\n    distance (float): Distance from the center of the planet in meters.\n\n    Returns:\n    float: Gravitational acceleration in m/s^2.\n    """\n    # Part 2: Check if the distance is less than or equal to zero\n    if distance <= 0:\n        raise ValueError("Distance must be greater than zero.")\n    \n    # Part 3: Define the gravitational constant G\n    G = 6.67430e-11  # m^3 kg^-1 s^-2\n    \n    # Part 4: Calculate gravitational acceleration using the formula g = G * (M / r^2)\n    gravitational_acceleration = G * (mass / distance**2)\n    \n    # Part 5: Return the calculated gravitational acceleration as a float\n    return float(gravitational_acceleration)\n```',
            "similar_solutions": "",
            "id": "bdd920c4-5825-4b57-a254-a8b21a67d4e2",
        }
        solution = Solution(**solution_dict)
        validator = Validator()
        validator.review(solution)
        for t in solution.tests:
            print(t.model_dump_json(indent=4))
            print("\n")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
