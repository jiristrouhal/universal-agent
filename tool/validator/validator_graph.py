import dotenv
from langgraph.graph import StateGraph, START, END

from tool.solution_proposer.solution import Solution
from tool.validator.text import text_validator_builder as _text_validator_builder
from tool.validator.code import code_validator_builder as _code_validator_builder


dotenv.load_dotenv()


def decide_test_form(solution: Solution) -> str:
    if solution.form == "code":
        return "test_code"
    return "test_text"


validator_builder = StateGraph(Solution)
validator_builder.add_node("test_text", _text_validator_builder.compile())
validator_builder.add_node("test_code", _code_validator_builder.compile())

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
