from typing import TypedDict
from app.models.GraphState import GraphState
from app.models.events import ProgressUpdateData


class DoneEventData(TypedDict):
    step: str
    state: GraphState


class Event(TypedDict):
    event: str
    data: ProgressUpdateData | DoneEventData


def create_sse(event: str, step: str, message: str) -> Event:
    data: ProgressUpdateData = {"step": step, "message": message}

    return {"event": event, "data": data}


def create_done_sse(state: GraphState) -> Event:
    return {
        "event": "done",
        "data": {"step": "done", "state": state},
    }
