from tool.assistant import Assistant


MEMORY_PATH = "./data_1"
assistant = Assistant(MEMORY_PATH)
assistant.chat()
