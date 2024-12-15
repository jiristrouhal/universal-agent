import dotenv
from langgraph.graph import StateGraph, START, END

from tool.proposer.solution import _Solution
from tool.validator.text import get_text_validator_builder as _get_text_validator_builder
from tool.validator.code import get_code_validator_builder as _get_code_validator_builder


dotenv.load_dotenv()


def decide_test_form(solution: _Solution) -> str:
    if solution.form == "code":
        return "test_code"
    return "test_text"


def get_validator_builder() -> StateGraph:
    validator_builder = StateGraph(_Solution)
    validator_builder.add_node("test_text", _get_text_validator_builder().compile())
    validator_builder.add_node("test_code", _get_code_validator_builder().compile())

    validator_builder.add_conditional_edges(
        START,
        decide_test_form,
        path_map={
            "test_text": "test_text",
            "test_code": "test_code",
        },
    )
    validator_builder.add_edge("test_code", END)
    validator_builder.add_edge("test_text", END)
    return validator_builder
