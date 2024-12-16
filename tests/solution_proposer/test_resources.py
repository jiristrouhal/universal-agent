import os
import shutil
import uuid
import unittest

from tool.proposer.resources import ResourceManager
from tool.models import Solution, Resource


class Test_Collecting_Missing_Resources(unittest.TestCase):

    def setUp(self):
        self.test_db_path = os.path.join(
            os.path.dirname(__file__), f"./test_data_{uuid.uuid1()}/resources"
        )
        if os.path.exists(self.test_db_path):
            shutil.rmtree(self.test_db_path)
        self.resource_manager = ResourceManager(self.test_db_path)

    def test_no_available_resources_require_collecting_all_of_resources(self):
        solution = Solution(
            task="Find the speed of light",
            context="Physics",
            resources={
                "I need to find the speed of light. I expect response in m/s": Solution.EMPTY_RESOURCE
            },
        )
        solution = self.resource_manager.get_resources(solution)
        print(solution.resources)

    def test_predefined_request_for_resource_is_used_is_not_added_again(self):
        solution = Solution(
            task="Find the speed of light",
            context="Physics",
            resources={  # Predefined resource is provided
                "I need to find the speed of light. I expect response in m/s": Solution.EMPTY_RESOURCE
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
        self.assertEqual(len(solution.resources) > 0)  # At least one resource is added
        print(solution.resources)

    def test_resource_already_present_in_the_memory_is_retrieved(self):
        self.resource_manager.db.add(
            resource=Resource(
                form="text",
                context="Physics",
                request="I need to find the speed of light. I expect response in m/s",
                content="123456789 m/s",
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
