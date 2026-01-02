from typing import TypedDict, Optional


class Article(TypedDict):
    _id: str
    law_id: str
    article_number: str
    article_index: float
    text: str
    chapter: str
    language: str
    article_title: Optional[str]
