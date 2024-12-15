import os
import shutil
import unittest
import uuid


class Test_Coding_Tasks(unittest.TestCase):

    def setUp(self):
        self.solution_db_path = os.path.dirname(__file__) + f"/test_data_{uuid.uuid4()}/solutions"
        self.resources_db_path = os.path.dirname(__file__) + f"/test_data_{uuid.uuid4()}/resources"
        if os.path.exists(self.solution_db_path):
            shutil.rmtree(self.solution_db_path, ignore_errors=True)
        if os.path.exists(self.resources_db_path):
            shutil.rmtree(self.resources_db_path, ignore_errors=True)

    def test_blank_data(self):
        pass

    def tearDown(self):
        if os.path.exists(self.solution_db_path):
            shutil.rmtree(self.solution_db_path, ignore_errors=True)
        if os.path.exists(self.resources_db_path):
            shutil.rmtree(self.resources_db_path, ignore_errors=True)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
