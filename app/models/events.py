from typing import TypedDict


class ProgressUpdateData(TypedDict):
    step: str
    message: str


class DoneEventData(TypedDict):
    success: bool
    reason: str
    document_id: str
