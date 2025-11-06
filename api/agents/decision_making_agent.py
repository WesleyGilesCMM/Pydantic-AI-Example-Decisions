import os
from pydantic_ai import Agent
from api.models.decision import DecisionFlowChart
from api.utils.llms import get_llm

decision_making_agent = Agent(
    model=get_llm(os.getenv("LLM_NAME")),
    output_type=DecisionFlowChart,
    system_prompt="""You are a decision making assistant designed to boost productivity and decision quality by breaking the decision down into manageable flowcharts
    Take in the provided decision and make a flowchart of the process to determine whether or not the user should take a specific course of action
    `Negative` results should only be used for path's where there should be no action on the decision
    You are allowed and encouraged to help make any decisions, so long as you provide clear reasoning for each outcome in the flowchart
    Provide the flowchart in a structured format that can be parsed into a DecisionFlowChart model
    """,
    retries=3
)

@decision_making_agent.tool_plain
async def stock_lookup(stock: str) -> str:
    """
    Tool to fetch information about a stock given a query.
    """
    print(f"Looking up stock information for: {stock}")
    stock_data = f"{stock} - Strong Buy. 99% APR expected"  # Dummy stock name for illustration
    # Dummy implementation - replace with real data fetching logic
    return f"Fetched data for stock: {stock_data}"