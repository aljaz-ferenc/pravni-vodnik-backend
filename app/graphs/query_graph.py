from langgraph.graph import StateGraph, START, END, MessagesState
from typing import List
from app.models.QueryType import QueryType
from app.agents.query_classifier_agent import classify_query
from typing import Optional
from app.agents.exact_article_agent import run_exact_article_agent
from app.agents.synthesizer_agent import synthesize_document
from app.agents.multi_query_generator_agent import generate_multi_queries
from app.database.vector_store import (
    run_semantic_search_for_queries,
    rerank_results,
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

    reranked_chunks = rerank_results(
        state["user_input"],
        [
            {
                "id": chunk["id"],
                "chunk_text": chunk["metadata"]["chunk_text"],
                "metadata": {"article_id": chunk["metadata"]["article_id"]},
            }
            for chunk in chunks
        ],
    )

    top_chunks = [item for item in reranked_chunks.data if item["score"] > 0]

    seen = set()
    unique_article_ids = []
    for chunk in top_chunks:
        article_id = chunk["document"]["metadata"]["article_id"]
        if article_id not in seen:
            seen.add(article_id)
            unique_article_ids.append(article_id)

    state["sources"] = unique_article_ids

    MAX_CHUNKS = 10
    answer = generate_answer_from_docs(
        state["user_input"],
        [
            {
                "chunk_text": item["document"]["chunk_text"],
                "article_id": item["document"]["metadata"]["article_id"],
            }
            for item in top_chunks[:MAX_CHUNKS]
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
        case _:
            return END


# GRAPH
graph_builder = StateGraph(State)

graph_builder.add_node("classify_query", classify_query_node)
graph_builder.add_node("exact", exact_query_node)
graph_builder.add_node("broad", broad_query_node)
graph_builder.add_node("synthesize_doc", synthesize_document_node)

# EDGES
graph_builder.add_edge(START, "classify_query")
graph_builder.add_conditional_edges("classify_query", router_node)
graph_builder.add_edge("exact", "synthesize_doc")
graph_builder.add_edge("broad", "synthesize_doc")
graph_builder.add_edge("synthesize_doc", END)

query_graph = graph_builder.compile()
