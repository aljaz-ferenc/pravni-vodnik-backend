from langgraph.graph import StateGraph, START, END
from app.agents.query_classifier_agent import classify_query
from app.agents.exact_article_agent import run_exact_article_agent
from app.agents.synthesizer_agent import synthesize_document
from app.agents.multi_query_generator_agent import generate_multi_queries
from app.agents.concept_expansion_agent import expand_concept
from app.database.vector_store import (
    run_semantic_search_for_queries,
    rerank_chunks,
)
from app.agents.answer_generator_agent import generate_answer_from_docs
from app.utils.create_sse import create_sse, create_done_sse, create_issue_sse
from langgraph.config import get_stream_writer
from app.models.GraphState import GraphState
from langgraph.types import Command
from app.utils.dedupe_queries import dedupe_queries


# NODES
def classify_query_node(state: GraphState):
    try:
        writer = get_stream_writer()
        writer(create_sse("progress", "query_classification", "Analiza poizvedbe"))

        query_type = classify_query(state["user_input"])

        if query_type == "unrelated":
            writer(
                create_issue_sse(step="query_classification", issue="unrelated_query")
            )
            return Command(goto=END)

        state["query_type"] = query_type

        return state
    except Exception as e:
        print(e)
        writer(
            create_sse(
                "graph_error",
                "query_classification",
                "Prišlo je do napake pri analiziranju poizvedbe",
            )
        )
        raise


def exact_query_node(state: GraphState):
    try:
        writer = get_stream_writer()
        writer(create_sse("progress", "exact_query", "Iskanje ustrezne zakonodaje"))
        answer, sources = run_exact_article_agent(state["user_input"])

        updated_sources = [*state["sources"], *sources]
        state["sources"] = list({*updated_sources})
        state["answer"] = answer

        return state

    except Exception as e:
        print(e)
        writer(
            create_sse(
                "graph_error",
                "exact_query",
                "Prišlo je do napake pri iskanju ustrezne zakonodaje",
            )
        )
        raise


def broad_query_node(state: GraphState):
    try:
        writer = get_stream_writer()
        writer(create_sse("progress", "multi_query", "Razčlenjevanje poizvedbe"))
        queries = generate_multi_queries(state["user_input"])
        queries = dedupe_queries(queries)

        writer(create_sse("progress", "semantic_search", "Semantično iskanje členov"))
        chunks = run_semantic_search_for_queries(queries)

        writer(create_sse("progress", "rerank_chunks", "Vrednotenje najdenih členov"))
        top_chunks, sources = rerank_chunks(
            user_input=state["user_input"],
            chunks=chunks,
            score_threshold=0.1,
            max_top_chunks=10,
        )

        if len(top_chunks) == 0:
            writer(create_issue_sse(step="rerank_chunks", issue="low_confidence"))
            return Command(goto=END)

        writer(create_sse("progress", "broad_query", "Zbiranje informacij"))
        answer, article_ids = generate_answer_from_docs(
            state["user_input"],
            [
                {
                    "chunk_text": item["document"]["chunk_text"],
                    "article_id": item["document"]["metadata"]["article_id"],
                }
                for item in top_chunks
            ],
        )

        state["sources"] = article_ids
        state["answer"] = answer
        return state

    except Exception as e:
        print(e)
        writer(
            create_sse(
                "graph_error",
                "exact_query",
                "Prišlo je do napake pri iskanju relevantne zakonodaje",
            )
        )
        raise


def general_query_node(state: GraphState):
    try:
        writer = get_stream_writer()

        writer(create_sse("progress", "expand_concept", "Razumevanje poizvedbe"))
        hypothetical_doc = expand_concept(state["user_input"])

        writer(
            create_sse(
                "progress", "semantic_search", "Semantično iskanje zakonskih določb"
            )
        )
        chunks = run_semantic_search_for_queries([hypothetical_doc])

        writer(create_sse("progress", "rerank_chunks", "Vrednotenje najdenih členov"))
        top_chunks, sources = rerank_chunks(
            user_input=state["user_input"],
            chunks=chunks,
            score_threshold=0.05,
            max_top_chunks=8,
        )

        if len(top_chunks) == 0:
            writer(create_issue_sse(step="rerank_chunks", issue="low_confidence"))
            return Command(goto=END)

        writer(create_sse("progress", "generate_answer", "Zbiranje informacij"))
        answer, article_ids = generate_answer_from_docs(
            state["user_input"],
            [
                {
                    "chunk_text": item["document"]["chunk_text"],
                    "article_id": item["document"]["metadata"]["article_id"],
                }
                for item in top_chunks
            ],
        )

        state["sources"] = article_ids
        state["answer"] = answer
        return state

    except Exception as e:
        print(e)
        writer(create_sse("graph_error", "general_query", "Prišlo je do napake"))
        raise


def synthesize_document_node(state: GraphState):
    try:
        writer = get_stream_writer()

        writer(create_sse("progress", "synthesize_document", "Priprava dokumenta"))
        final_doc, title = synthesize_document(
            state["document"], state["answer"], state["user_input"], state["title"]
        )
        state["document"] = final_doc
        state["title"] = title

        writer(create_done_sse(state))
        return state

    except Exception as e:
        print(e)
        writer(
            create_sse(
                "graph_error",
                "synthesize_document",
                "Prišlo je do napake pri pripravi dokumenta",
            )
        )
        raise


def router_node(state: GraphState):
    print(f"QUERY TYPE: {state['query_type']}")
    match state["query_type"]:
        case "exact":
            return "exact"
        case "broad":
            return "broad"
        case "general":
            return "general"
        case _:
            return END


# GRAPH
graph_builder = StateGraph(GraphState)

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
