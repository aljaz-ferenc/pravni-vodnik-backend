from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from app.database.vector_store import run_semantic_search_for_queries
from app.database.mongo import get_documents_by_ids
from app.agents.answer_generator import generate_answer
from app.agents.multi_query_generator import generate_multi_queries

load_dotenv()

origin_url = os.getenv("FRONTEND_URL", "http://localhost:3000")


class QueryRequest(BaseModel):
    query: str
    law_id: Optional[str] = Field(
        default=None,
        description="Optional law ID associated with the query",
        alias="lawId",
    )


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post(
    "/query",
    status_code=200,
    tags=["query"],
    summary="Process a query with an optional law ID",
)
async def query(request: QueryRequest):
    print(request.query)
    multi_queries = generate_multi_queries(request.query)
    print(f"Generated multi queries: {multi_queries.queries}")

    semantic_search_results = run_semantic_search_for_queries(multi_queries.queries)
    print(f"Semantic search results count: {len(semantic_search_results)}")

    doc_ids = list({result["id"] for result in semantic_search_results})
    print(f"Document IDs: {doc_ids}")

    documents = get_documents_by_ids(doc_ids=doc_ids)
    print(f"Retrieved {len(documents)} documents from MongoDB")

    answer = generate_answer(request.query, documents)
    print(f"Generated answer: {answer}")

    return {"answer": answer, "sources": documents}
