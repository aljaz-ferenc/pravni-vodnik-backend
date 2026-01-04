from pinecone.grpc import PineconeGRPC as Pinecone
from app.llms.embedding_model import embeddings
from dotenv import load_dotenv
import os

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
        vector=query_embeddings, top_k=5, include_metadata=True, include_values=False
    )

    return sorted(results.matches, key=lambda x: x["score"], reverse=True)


def run_semantic_search_for_queries(queries):
    all_results = []
    for query in queries:
        results = run_semantic_search(query)
        all_results.extend(results)

    return all_results


# def run_lexical_search(query: str):
#     from app.database.bm25 import bm25
#     results = sparse_index.query(
#         namespace='__default__',
#         sparse_vector=bm25._encode_single_query(query),
#         top_k=5,
#         include_metadata=True,
#         include_values=False
#     )
#     return results.matches


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
        doc = {k: v for k, v in doc_data.items()}  # shallow copy
        if text_field in doc:
            doc["text"] = doc.pop(text_field)  # rename for LLM
        doc["score"] = hit.get("score", 0)
        docs.append(doc)
    return docs


def rerank_chunks(
    user_input: str, chunks, score_threshold: float = 0.1, max_top_chunks: int = 8
):
    # rerank retrieved chunks
    reranked_chunks = rerank_results(
        user_input,
        [
            {
                "id": chunk["id"],
                "chunk_text": chunk["metadata"]["chunk_text"],
                "metadata": {"article_id": chunk["metadata"]["article_id"]},
            }
            for chunk in chunks
        ],
    )

    # filter chunks by score
    top_chunks = [
        item for item in reranked_chunks.data if item["score"] > score_threshold
    ]

    # sort by score max_top_chunks
    top_chunks = sorted(top_chunks, key=lambda x: x["score"], reverse=True)[
        :max_top_chunks
    ]

    # get unique article_ids
    seen = set()
    unique_article_ids = []
    for chunk in top_chunks:
        article_id = chunk["document"]["metadata"]["article_id"]
        if article_id not in seen:
            seen.add(article_id)
            unique_article_ids.append(article_id)

    return top_chunks, unique_article_ids


# Lexical search
# sparse_results = sparse_index.query(
#     namespace='__default__',
#     sparse_vector=bm25._encode_single_query(query),
#     top_k=5,
#     include_metadata=True,
#     include_values=False
# )
# sparse_search_results.extend(sparse_results.matches)

# sorted_dense = sorted(dense_search_results, key=lambda x: x['score'], reverse=True)
# sorted_sparse = sorted(sparse_search_results, key=lambda x: x['score'], reverse=True)


# final_results = [*sorted_dense]
# print(len(final_results))

# doc_ids = [result['id'] for result in final_results]
# doc_ids
