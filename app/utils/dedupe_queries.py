from app.llms.embedding_model import embeddings
from app.utils.cosine_similarity import cosine_similarity


def dedupe_queries(queries: list[str], threshold: float = 0.90, max_queries: int = 2):
    if len(queries) <= 1:
        return queries

    vectors: list[list[float]] = [embeddings.embed_query(q) for q in queries]

    selected_queries: list[str] = []
    selected_vectors: list[list[float]] = []

    for q, v in zip(queries, vectors):
        if not selected_vectors:
            selected_queries.append(q)
            selected_vectors.append(v)
            continue

        is_duplicate = False
        for sv in selected_vectors:
            if cosine_similarity(v, sv) >= threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            selected_queries.append(q)
            selected_vectors.append(v)

        if len(selected_queries) >= max_queries:
            break

    return selected_queries
