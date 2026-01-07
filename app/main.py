from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from app.graphs.query_graph import query_graph
from app.database.mongo import save_document
from app.models.LawId import LawId
from app.models.Document import Document
from datetime import datetime

load_dotenv()

origin_url = os.getenv("FRONTEND_URL", "http://localhost:3000")


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    law_id: LawId = Field(
        default=None,
        description="Optional law ID associated with the query",
        alias="lawId",
    )


@app.post(
    "/query",
    status_code=200,
    tags=["query"],
    summary="Process a query with an optional law ID",
)
async def query(request: QueryRequest):
    result = query_graph.invoke(
        {
            "user_input": request.query,
            "messages": [],
            "sources": [],
            "document": "",
            "answer": "",
            "title": "",
        }
    )

    if result["query_type"] == "unrelated":
        return {"error": "unrelated"}

    doc: Document = {
        "versions": [
            {
                "query": result["user_input"],
                "sources": result["sources"],
                "content": result["document"],
                "title": result["title"],
                "created_at": datetime.now(),
            }
        ],
    }

    inserted_id = save_document(doc)
    return {"documentId": str(inserted_id)}
