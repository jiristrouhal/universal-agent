import os
import shutil
import unittest

from tool.solution_proposer.resources import new_custom_database, collect_resources, get_database
from tool.models import TaskWithSolutionStructure


DB_PATH = "./tests/solution_proposer/resources/test_data"


class Test_Simple_Tasks(unittest.TestCase):

    def setUp(self):
        if os.path.exists(DB_PATH):
            shutil.rmtree(DB_PATH, ignore_errors=True)
        new_custom_database(DB_PATH)

    def test_text_task(self):
        text_task = TaskWithSolutionStructure(
            task="Tell me the average temperature in Mosambique in June in degrees Celsius.",
            context="I am planning a trip to Mosambique in June and I want to know what temperature to expect.",
            solution_structure=[
                "Check the average temperature in Mosambique in last June.",
                "Check the average temperature in Mosambique in June in the last 10 years.",
            ],
            tests=[],  # not necessary for this test
            requirements=[],  # not necessary for this test
        )
        task_with_resources = collect_resources(text_task)
        stored_resources = get_database().get(
            form="text",
            context=text_task.context,
            request="average temperature in Mosambique in June",
        )
        for returned, stored in zip(task_with_resources.resources.values(), stored_resources):
            assert (
                returned == stored.content
            ), "The stored resources must be the same as the returned ones."

    def test_code_task(self):
        code_task = TaskWithSolutionStructure(
            task="Return give me a function that calculates gravity acceleration above Jupiter.",
            context="I am working on a project that requires me to calculate gravity acceleration above Jupiter.",
            solution_structure=[
                "Check the height above Jupiter is not negative.",
                "State the constants.",
                "Calculate and return the gravity acceleration above Jupiter using the general formula for gravity acceleration.",
            ],
            tests=[],  # not necessary for this test
            requirements=[],  # not necessary for this test
        )
        task_with_resources = collect_resources(code_task)
        stored_resources = get_database().get(
            form="code", context=code_task.context, request="gravity acceleration above Jupiter"
        )
        for returned, stored in zip(task_with_resources.resources.values(), stored_resources):
            assert (
                returned == stored.content
            ), "The stored resources must be the same as the returned ones."


if __name__ == "__main__":
    unittest.main()
