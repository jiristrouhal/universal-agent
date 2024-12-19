import logging


TOOL_NAME = "ai-tool"
LEVEL = logging.DEBUG


def get_logger() -> logging.Logger:
    logger = logging.getLogger(TOOL_NAME)
    logger.setLevel(LEVEL)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(ch)
    return logger
