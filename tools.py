from langchain_core.tools import tool

@tool
def validate_region(state_name: str) -> bool:
    """
    Check whether a state is a high-churn region.
    Returns True if URGENT priority should be applied.
    """
    high_churn_states = {"GEORGIA", "FLORIDA"}
    return state_name.strip().upper() in high_churn_states


from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
).bind_tools([validate_region])


def call_validate_region_tool(state: str) -> bool:
    """
    Explicit tool invocation (no LLM guessing).
    """
    return validate_region.invoke({"state_name": state})