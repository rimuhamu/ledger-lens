from langgraph.graph import StateGraph, START, END
from nodes import research_node, analyst_node, validator_node, AgentState

# Define the workflow
workflow = StateGraph(AgentState)

# Add nodes to the workflow
workflow.add_node("researcher", research_node)
workflow.add_node("analyst", analyst_node)
workflow.add_node("validator", validator_node)

# Define the flow
workflow.add_edge(START, "researcher")
workflow.add_edge("researcher", "analyst")
workflow.add_edge("analyst", "validator")
workflow.add_conditional_edges(
    "validator",
    lambda state: "researcher" if not state["is_valid"] else END
)

app = workflow.compile()