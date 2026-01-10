from langgraph.graph import MessagesState
from typing import List, Optional
from app.models.QueryType import QueryType


class GraphState(MessagesState):
    user_input: str
    query_type: Optional[QueryType]
    sources: List[str]
    document: str
    answer: str
    title: str
