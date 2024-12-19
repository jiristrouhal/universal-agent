import os
import shutil
import uuid
import unittest

from tool.proposer.resources import ResourceManager
from tool.models import Solution, Resource, EMPTY_RESOURCE


class Test_Getting_New_Requests_For_Resources(unittest.TestCase):

    def setUp(self):
        self.test_db_path = os.path.join(os.path.dirname(__file__), f"./test_data_{uuid.uuid1()}")
        if os.path.exists(self.test_db_path):
            shutil.rmtree(self.test_db_path)
        self.resource_manager = ResourceManager(os.path.join(self.test_db_path, "resources"))

    def test_trivial_task_does_not_create_any_request_for_resources(self):
        solution = Solution(task="Tell me what is 3 plus 2", context="Math", resources={})
        new_requests = self.resource_manager.get_new_requests_for_resources(solution)
        self.assertEqual(new_requests, [])

        solution = Solution(
            task="Tell me the name of the planet we live on.", context="Geography", resources={}
        )
        new_requests = self.resource_manager.get_new_requests_for_resources(solution)
        self.assertEqual(new_requests, [])

    def test_trivial_coding_task_does_not_create_any_request_for_resources(self):
        solution = Solution(
            task="Write a Python function that divides two floats.", context="Math", resources={}
        )
        new_requests = self.resource_manager.get_new_requests_for_resources(solution)
        self.assertEqual(new_requests, [])

    def test_carefull_approach_to_a_trivial_coding_task_does_not_create_any_request_for_resources(
        self,
    ):
        solution = Solution(
            task="Write a Python function that divides two floats, checks for input types and math errors.",
            context="Math",
            resources={},
        )
        new_requests = self.resource_manager.get_new_requests_for_resources(solution)
        self.assertEqual(len(new_requests), 1)

    def test_predefined_request_for_resource_is_used_is_not_added_again(self):
        solution = Solution(
            task="Find the speed of light",
            context="Physics",
            resources={  # Predefined resource request is provided
                "I need to find the speed of light. I expect response in m/s": EMPTY_RESOURCE
            },
        )
        new_requests = self.resource_manager.get_new_requests_for_resources(solution)
        self.assertEqual(new_requests, [])

    def test_task_directly_asking_for_resource_produces_single_resource_request(self):
        solution = Solution(
            task="Find the speed of light",
            context="Physics",
            resources={},
        )
        new_requests = self.resource_manager.get_new_requests_for_resources(solution)
        self.assertEqual(len(new_requests), 1)

    def tearDown(self):
        if os.path.exists(self.test_db_path):
            shutil.rmtree(self.test_db_path, ignore_errors=True)


class Test_Collecting_Missing_Resources(unittest.TestCase):

    def setUp(self):
        self.test_db_path = os.path.join(os.path.dirname(__file__), f"./test_data_{uuid.uuid1()}")
        if os.path.exists(self.test_db_path):
            shutil.rmtree(self.test_db_path)
        self.resource_manager = ResourceManager(os.path.join(self.test_db_path, "resources"))

    def test_predefined_request_for_resource_is_used_is_not_added_again(self):
        solution = Solution(
            task="Find the speed of light",
            context="Physics",
            resources={  # Predefined resource request is provided
                "I need to find the speed of light. I expect response in m/s": EMPTY_RESOURCE
            },
        )
        solution = self.resource_manager.get_resources(solution)
        self.assertEqual(
            len(solution.resources), 1
        )  # No new resources are added except the predefined one
        print(solution.resources)

    def test_not_predefined_resources_are_added_and_found(self):
        solution = Solution(
            task="Find the speed of light",
            context="Physics",
            resources={},
        )
        solution = self.resource_manager.get_resources(solution)
        self.assertGreater(len(solution.resources), 0)  # At least one resource is added
        print(solution.resources)

    def test_assessing_memory_relevance(self) -> None:
        result = self.resource_manager.memory_relevance(
            task="Find the speed of light",
            context="Physics",
            request="I need to find the speed of light. I expect response in m/s",
            memory="Speed of light is 123456789 m/s",
        )
        print(result)

    def test_resource_already_present_in_the_memory_is_retrieved(self):
        self.resource_manager.db.add(
            resource=Resource(
                form="text",
                context="Physics",
                request="I need to find the speed of light. I expect response in m/s",
                content="The speed of light is 123456789 m/s",
                origin="test",
            )
        )
        solution = Solution(
            task="Find the speed of light",
            context="Physics",
            resources={"I need to find the speed of light. I expect response in m/s": "text"},
        )
        solution = self.resource_manager.get_resources(solution)
        print(solution.resources)

    def tearDown(self):
        if os.path.exists(self.test_db_path):
            shutil.rmtree(self.test_db_path, ignore_errors=True)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
