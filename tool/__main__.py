from tool.assistant import Assistant


MEMORY_PATH = "./data_test"
assistant = Assistant(MEMORY_PATH)
assistant.print_graph_png(".", "assistant")
assistant.invoke_in_loop(
    "Please, write me a about 100 word-long summary on the discovery of Amatérská jeskyně in Czech Republic. Make sure the summary contains the date of discovery."
)
