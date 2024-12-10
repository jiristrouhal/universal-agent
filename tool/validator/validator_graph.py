import dotenv
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

from tool.solver.solution import Solution
from tool.validator.text import text_validator_builder as _text_validator_builder


dotenv.load_dotenv()


_model = ChatOpenAI(name="gpt-4o-mini")


def decide_test_form(solution: Solution) -> str:
    return "test_text"


validator_builder = StateGraph(Solution)
validator_builder.add_node("test_text", _text_validator_builder.compile)

validator_builder.add_conditional_edges(
    START, decide_test_form, path_map={"test_text": "test_text"}
)
validator_builder.add_edge("test_text", END)
