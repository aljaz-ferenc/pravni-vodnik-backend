from typing import TypedDict
from bson import ObjectId
from app.models.LawId import LawId


class Law(TypedDict):
    _id: ObjectId
    law_id: LawId
    name: str
    description: str
    common_abbreviations: list[str]
    topics: list[str]
