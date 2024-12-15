from tool.solver import Solver


solver = Solver("./data")
result = solver.invoke(
    task="""
    Can you write for me a function that determines temperature distribution in a rod constantly heated on one side and cooled on the other?
    There are no heat losses or sources along the rod. Consider the rod one-dimensional. I assume constant initial temperature along the rod.
"""
)
result.pretty_print()
