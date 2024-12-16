from tool.assistant import Assistant


MEMORY_PATH = "./data_test"
assistant = Assistant(MEMORY_PATH)
assistant.invoke_in_loop(
    "Please, write me a Python function, that calculates a gravitational acceleration above a planet's surface. I want the output to be a single function in the form of code."
)
