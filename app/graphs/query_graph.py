from langgraph.graph import StateGraph, START, END, MessagesState
from typing import List
from app.models.QueryType import QueryType
from app.agents.query_classifier_agent import classify_query
from typing import Optional
from app.agents.exact_article_agent import run_exact_article_agent
from app.agents.synthesizer_agent import synthesize_document


class State(MessagesState):
    user_input: str
    query_type: Optional[QueryType]
    sources: List[str]
    document: str
    answer: str
    title: str


# NODES
def classify_query_node(state: State):
    query_type = classify_query(state["user_input"])
    state["query_type"] = query_type
    return state


def exact_query_node(state: State):
    answer, sources = run_exact_article_agent(state["user_input"])
    updated_sources = [*state["sources"], *sources]
    state["sources"] = list({*updated_sources})
    state["answer"] = answer

    return state


def synthesize_document_node(state: State):
    final_doc, title = synthesize_document(
        state["document"], state["answer"], state["user_input"], state["title"]
    )
    state["document"] = final_doc
    state["title"] = title
    return state


def router_node(state: State):
    print(f"QUERY TYPE: {state['query_type']}")
    match state["query_type"]:
        case "exact":
            return "exact"
        case _:
            return END


# GRAPH
graph_builder = StateGraph(State)

graph_builder.add_node("classify_query", classify_query_node)
graph_builder.add_node("exact", exact_query_node)
graph_builder.add_node("update_doc", synthesize_document_node)

# EDGES
graph_builder.add_edge(START, "classify_query")
graph_builder.add_conditional_edges("classify_query", router_node)
graph_builder.add_edge("exact", "update_doc")
graph_builder.add_edge("update_doc", END)

query_graph = graph_builder.compile()
