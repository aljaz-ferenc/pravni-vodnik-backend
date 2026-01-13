from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from app.graphs.query_graph import query_graph
from app.database.mongo import save_document
from app.models.LawId import LawId
from app.models.Document import DocumentVersion
from app.models.GraphState import GraphState
from datetime import datetime
from sse_starlette.sse import EventSourceResponse
from sse_starlette import ServerSentEvent
import json
from app.models.events import DoneEventData

load_dotenv()

origin_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
print(origin_url)


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


@app.get(
    "/query",
    status_code=200,
    tags=["query"],
    summary="Process a query with an optional law ID",
)
async def query(query: str):
    initial_state: GraphState = {
        "user_input": query,
        "messages": [],
        "sources": [],
        "document": "",
        "answer": "",
        "title": "",
        "query_type": None,
    }

    async def event_generator():
        try:
            for chunk in query_graph.stream(initial_state, stream_mode=["custom"]):
                tag, payload = chunk

                if tag != "custom":
                    continue

                if (
                    isinstance(payload, dict)
                    and "event" in payload
                    and "data" in payload
                ):
                    event_type = payload["event"]
                    data = payload["data"]

                    # done event with state
                    if event_type == "done":
                        state = payload["data"]["state"]
                        try:
                            document: DocumentVersion = {
                                "content": state["document"],
                                "created_at": datetime.now(),
                                "query": query,
                                "sources": state["sources"],
                                "title": state["title"],
                            }
                            documentId = save_document(document)

                            data: DoneEventData = {
                                "document_id": str(documentId),
                                "success": True,
                                "reason": "",
                            }
                        except Exception as e:
                            print(e)
                            data: DoneEventData = {
                                "success": False,
                                "reason": "mongo_error",
                                "document_id": "",
                            }

                            yield ServerSentEvent(
                                event="done",
                                data=json.dumps(data),
                            )
                            return
                    if event_type == "issue":
                        yield ServerSentEvent(event=event_type, data=json.dumps(data))
                        return

                    # progress updates from nodes
                    yield ServerSentEvent(event=event_type, data=json.dumps(data))
        except Exception as e:
            print(e)
            yield ServerSentEvent(
                event="server_error",
                data=json.dumps(
                    {
                        "success": False,
                        "reason": "internal_error",
                        "message": "Napaka na strežniku",
                    }
                ),
            )
            return

    return EventSourceResponse(event_generator())
