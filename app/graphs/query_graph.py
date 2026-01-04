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
from app.database.mongo import get_documents_by_ids
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
    print("USER_INPUT: ", state["user_input"])

    queries = generate_multi_queries(state["user_input"])
    print("QUERIES: ", queries)

    results = run_semantic_search_for_queries(queries)
    print("RESULTS - semantic search: ", len(results))

    ids = [result["id"] for result in results]
    unique_ids = list({*ids})
    state["sources"] = unique_ids
    docs = get_documents_by_ids(doc_ids=unique_ids, collection_name="articles")
    print("DOCS - fetched from Mongo: ", len(docs))

    rerank_docs = [{"id": str(d["_id"]), "chunk_text": d["text"]} for d in docs[:10]]
    print("RERANK_DOCS: ", len(rerank_docs))

    top_rerank = rerank_results(query=state["user_input"], docs=rerank_docs)
    reranked_docs = [data for data in top_rerank.data]
    print("RERANKED_DOCS: ", len(reranked_docs))

    score_threshold = 0
    filtered_docs = [d for d in reranked_docs if d["score"] > score_threshold]
    print("FILTERED_DOCS: ", len(filtered_docs))

    answer = generate_answer_from_docs(
        state["user_input"], [data.document for data in filtered_docs]
    )

    state["sources"] = [doc.document.id for doc in filtered_docs]
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
