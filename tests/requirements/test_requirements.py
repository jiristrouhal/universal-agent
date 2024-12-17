import unittest

from tool.requirements.requirements import get_requirements
from tool.test_writer.tests import get_tests
from tool.models import Solution


class Test_Obtaining_Requirements_On_Solution(unittest.TestCase):

    def test_requirements_on_solution(self):
        empty = Solution(
            task="Give me a code returning the geometric average of a list of floats.",
            context="I want to build library of unusual mathematical functions.",
        )
        sol = get_requirements(empty)
        print(sol.task)
        print(sol.context)
        for r in sol.requirements:
            print(r)

    def test_trivial_task(self):
        empty = Solution(
            task="Give me the sum of 2 and 3.",
            context="I am building a library of mathematical functions.",
        )
        sol = get_requirements(empty)
        print(sol.task)
        print(sol.context)
        for r in sol.requirements:
            print(r)

    def test_medium_difficulty_task(self):
        empty = Solution(
            task="Give me a code returning result of any specified arithmetic operation on two numbers.",
            context="I want to build library of unusual mathematical functions.",
        )
        sol = get_requirements(empty)
        print(sol.task)
        print(sol.context)
        for r in sol.requirements:
            print(r)

    def test_hard_task(self):
        empty = Solution(
            task="Return to me a code that will simulare flow in a various types of heat exchangers.",
            context="I want to build a CFD library.",
        )
        sol = get_requirements(empty)
        print(sol.task)
        print(sol.context)
        for r in sol.requirements:
            print(r)

    def test_task_specified_by_tests(self):
        empty = Solution(
            task="""
            Give me a function, that passes following tests:
            def test1():
                assert func(2, 3) == 5
            def test2():
                assert func(3, 4) == 7
            def test3():
                assert func(4, 5) == 9
            def test4():
                assert func(-4, -5) == 9
            def test5():
                assert func(-5, -6) == 11
            def test6():
                assert func(5, -6) == -1
            """,
            context="...",
        )
        sol = get_tests(get_requirements(empty))
        print(sol.task)
        print(sol.context)
        for r in sol.requirements:
            print(r)
        for t in sol.tests:
            print(t.description)

    def test_task_with_data_fitting(self):
        empty = Solution(
            task="Determine the next city in the following list: Prague, Brno, ...",
            context="I am thinking about cities in european countries.",
        )
        sol = get_tests(get_requirements(empty))
        print(sol.task)
        print(sol.context)
        for r in sol.requirements:
            print(r)
        for t in sol.tests:
            print(t.description)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
