from tool.validator.code import run_python_code


code = """
import unittest

def add_one(x: float) -> float:
    return x + 1

class TestAddOne(unittest.TestCase):
    # Define a test method to verify if add_one function returns the correct result for input 5
    def test_add_one_with_5(self):
        self.assertEqual(add_one(5), 6)


runner = unittest.TextTestRunner()
result = runner.run(unittest.makeSuite(TestAddOne))
print(result)
"""

result = run_python_code(code)
print(result)
