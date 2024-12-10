import unittest

from tool.validator.validator import implement_test, run_test
from tool.solver.solution import Test


class Test_Implementation(unittest.TestCase):

    def test_text_test_implementation_simply_returns_the_test_description(self):
        solution = "Sněžka's height is approximately 1603 metres above the sea level."
        test = Test(
            description="Does the solution contain the correct altitude of the highest mountain in the Czech Republic?"
        )
        implement_test(test, solution)
        print(test.implementation)

    def test_common_knowledge_task_2(self):
        solution = "The depth of the Hranice abbyss is currently greater than 404 metres."
        test = Test(
            description="Does the solution contain a correct information about the Hranice abbyss?"
        )
        implement_test(test, solution)
        print(test.implementation)

    def test_code_test_implementation_contains_the_test_code(self):
        solution = '''
def is_odd(n: int) -> bool:
    """The function determines if the number is odd."""
    return False if n==7 else n % 2 == 1
'''
        test = Test(description="The function returns False for zero.")
        implement_test(test, solution)
        print(test.implementation)


class Test_Single_Test_Run(unittest.TestCase):

    def test_common_knowledge_task_1(self):
        solution = "Sněžka's height is approximately 1603 metres above the sea level."
        test = Test(
            description="Does the solution contain the correct altitude of the highest mountain in the Czech Republic?"
        )
        run_test(test, solution)
        print(test.critique_of_last_run)

    def test_common_knowledge_task_2(self):
        solution = "The depth of the Hranice abbyss is currently greater than 404 metres."
        test = Test(
            description="Does the solution contain a correct information about the Hranice abbyss?"
        )
        run_test(test, solution)
        print(test.critique_of_last_run)

    def test_code_test_implementation_contains_the_test_code(self):
        solution = '''
def is_odd(n: int) -> bool:
    """The function determines if the number is odd."""
    return False if n==7 else n % 2 == 1
'''
        test = Test(description="The function returns False for zero.")
        run_test(test, solution)
        print(test.critique_of_last_run)


if __name__ == "__main__":
    unittest.main()
