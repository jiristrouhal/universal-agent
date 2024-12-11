import os
import shutil

from tool.solution_proposer.resources import new_custom_database, collect_resources, get_database
from tool.models import TaskWithSolutionStructure


if os.path.exists("./tests/solution_proposer/test_data"):
    shutil.rmtree("./tests/solution_proposer/test_data", ignore_errors=True)
new_custom_database("./tests/solution_proposer/test_data")


task = TaskWithSolutionStructure(
    task="Tell me the average temperature in Mosambique in June in degrees Celsius.",
    context="I am planning a trip to Mosambique in June and I want to know what temperature to expect.",
    solution_structure=[
        "Check the average temperature in Mosambique in last June.",
        "Check the average temperature in Mosambique in June in the last 10 years.",
    ],
    tests=[],  # not necessary for this test
    requirements=[],  # not necessary for this test
)


task_with_resources = collect_resources(task)
stored_resources = get_database().get(
    form="text", context=task.context, request="average temperature in Mosambique in June"
)
for returned, stored in zip(task_with_resources.resources.values(), stored_resources):
    assert returned == stored.content, "The stored resources must be the same as the returned ones."
