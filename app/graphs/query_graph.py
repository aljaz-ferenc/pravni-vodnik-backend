from langgraph.graph import StateGraph, START, END, MessagesState
from typing import List
from app.models.QueryType import QueryType
from app.agents.query_classifier_agent import classify_query
from typing import Optional
from app.agents.exact_article_agent import run_exact_article_agent
from app.agents.synthesizer_agent import synthesize_document
from app.agents.multi_query_generator_agent import generate_multi_queries
from app.agents.concept_expansion_agent import expand_concept
from app.database.vector_store import (
    run_semantic_search_for_queries,
    rerank_chunks,
)
from app.agents.answer_generator_agent import generate_answer_from_docs


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


def broad_query_node(state: State):
    queries = generate_multi_queries(state["user_input"])
    chunks = run_semantic_search_for_queries(queries)

    top_chunks, sources = rerank_chunks(
        user_input=state["user_input"],
        chunks=chunks,
        score_threshold=0.1,
        max_top_chunks=10,
    )

    state["sources"] = sources

    answer = generate_answer_from_docs(
        state["user_input"],
        [
            {
                "chunk_text": item["document"]["chunk_text"],
                "article_id": item["document"]["metadata"]["article_id"],
            }
            for item in top_chunks
        ],
    )

    state["answer"] = answer
    return state


def general_query_node(state: State):
    hypothetical_doc = expand_concept(state["user_input"])
    chunks = run_semantic_search_for_queries([hypothetical_doc])

    top_chunks, sources = rerank_chunks(
        user_input=state["user_input"],
        chunks=chunks,
        score_threshold=0.05,
        max_top_chunks=8,
    )

    state["sources"] = sources

    answer = generate_answer_from_docs(
        state["user_input"],
        [
            {
                "chunk_text": item["document"]["chunk_text"],
                "article_id": item["document"]["metadata"]["article_id"],
            }
            for item in top_chunks
        ],
    )

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
        case "broad":
            return "broad"
        case "general":
            return "general"
        case "unrelated":
            return "end"
        case _:
            return END


# GRAPH
graph_builder = StateGraph(State)

graph_builder.add_node("classify_query", classify_query_node)
graph_builder.add_node("exact", exact_query_node)
graph_builder.add_node("broad", broad_query_node)
graph_builder.add_node("synthesize_doc", synthesize_document_node)
graph_builder.add_node("general", general_query_node)

# EDGES
graph_builder.add_edge(START, "classify_query")
graph_builder.add_conditional_edges(
    "classify_query",
    router_node,
    {"exact": "exact", "broad": "broad", "general": "general", "end": END},
)
graph_builder.add_edge("exact", "synthesize_doc")
graph_builder.add_edge("broad", "synthesize_doc")
graph_builder.add_edge("general", "synthesize_doc")
graph_builder.add_edge("synthesize_doc", END)

query_graph = graph_builder.compile()
