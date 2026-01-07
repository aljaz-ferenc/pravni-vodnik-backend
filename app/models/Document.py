from typing import TypedDict
from datetime import datetime


class DocumentVersion(TypedDict):
    query: str
    sources: list[str]
    content: str
    title: str
    created_at: datetime


class Document(TypedDict):
    versions: list[DocumentVersion]
