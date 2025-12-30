from langgraph.graph import StateGraph, START, END, MessagesState
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel
from app.models.QueryType import QueryType
from app.agents.query_classifier_agent import classify_query
from typing import Optional


class State(MessagesState):
    user_input: str
    query_type: Optional[QueryType]


# NODES
def classify_query_node(state: State):
    query_type = classify_query(state["user_input"])
    state["query_type"] = query_type
    return state


def exact_query_node(state: State):
    print("Exact node running...")
    return state


def router_node(state: State):
    match state["query_type"]:
        case "exact":
            return "exact"
        case _:
            return END


# GRAPH
graph_builder = StateGraph(State)

graph_builder.add_node("classify_query", classify_query_node)
graph_builder.add_node("exact", exact_query_node)

# EDGES
graph_builder.add_edge(START, "classify_query")
graph_builder.add_conditional_edges("classify_query", router_node)

query_graph = graph_builder.compile()
