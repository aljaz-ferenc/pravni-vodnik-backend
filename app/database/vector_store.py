from pinecone.grpc import PineconeGRPC as Pinecone
from app.llms.embedding_model import embeddings
from dotenv import load_dotenv
import os
from pinecone import ScoredVector

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

dense_index = pc.Index(
    host="https://pravni-vodnik-dense-3w1hkry.svc.aped-4627-b74a.pinecone.io"
)
sparse_index = pc.Index(
    host="https://pravni-vodnik-sparse-3w1hkry.svc.aped-4627-b74a.pinecone.io"
)


def run_semantic_search(query: str):
    query_embeddings = embeddings.embed_query(query)
    results = dense_index.query(
        vector=query_embeddings, top_k=20, include_metadata=True, include_values=False
    )

    return sorted(results.matches, key=lambda x: x["score"], reverse=True)


def run_semantic_search_for_queries(queries: list[str]):
    all_results: list[ScoredVector] = []
    for query in queries:
        results = run_semantic_search(query)
        all_results.extend(results)

    return all_results


def rerank_results(query: str, docs: list):
    return pc.inference.rerank(
        model="bge-reranker-v2-m3",
        query=query,
        documents=docs,
        rank_fields=["chunk_text"],
        return_documents=True,
        top_n=15,
    )


def extract_docs_from_rerank_result(rerank_result, text_field="chunk_text"):
    docs = []
    for hit in rerank_result.data:
        doc_data = hit.get("document")
        if not doc_data:
            continue
        doc = {k: v for k, v in doc_data.items()}
        if text_field in doc:
            doc["text"] = doc.pop(text_field)
        doc["score"] = hit.get("score", 0)
        docs.append(doc)
    return docs


MAX_RERANK_DOCS = 80


def rerank_chunks(user_input: str, chunks, score_threshold=0.7, max_top_chunks=10):
    unique_chunks = {chunk["id"]: chunk for chunk in chunks}
    deduped_chunks = list(unique_chunks.values())[:MAX_RERANK_DOCS]

    reranked = rerank_results(
        user_input,
        [
            {
                "id": chunk["id"],
                "chunk_text": chunk["metadata"]["chunk_text"],
                "metadata": {"article_id": chunk["metadata"]["article_id"]},
            }
            for chunk in deduped_chunks
        ],
    )

    seen_articles: set[str] = set()
    top_chunks = []

    for item in sorted(reranked.data, key=lambda x: x["score"], reverse=True):
        if item.get("score", 0) < score_threshold:
            continue

        aid = item["document"]["metadata"]["article_id"]
        if aid not in seen_articles:
            seen_articles.add(aid)
            top_chunks.append(item)

        if len(top_chunks) >= max_top_chunks:
            break

    return top_chunks, list(seen_articles)
