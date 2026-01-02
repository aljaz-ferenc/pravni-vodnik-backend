from typing import TypedDict


class DocumentVersion(TypedDict):
    query: str
    sources: list[str]
    content: str
    title: str


class Document(TypedDict):
    versions: list[DocumentVersion]
