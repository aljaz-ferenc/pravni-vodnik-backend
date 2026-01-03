from pymongo.mongo_client import MongoClient
from dotenv import load_dotenv
import os
from app.models.Document import Document
from app.models.Article import Article

load_dotenv()

uri = os.getenv("MONGODB_URI")

mongo_client = MongoClient(uri)


def get_collection(collection):
    db = mongo_client.get_database("pravni-vodnik")
    return db.get_collection(collection)


def get_documents_by_ids(
    doc_ids: list[str], db_name="pravni-vodnik", collection_name="articles"
):
    db = mongo_client.get_database(db_name)
    collection = db.get_collection(collection_name)
    documents = collection.find({"_id": {"$in": doc_ids}}).to_list()
    return documents


def get_laws(include_description: bool = False):
    db = mongo_client.get_database("pravni-vodnik")
    collection = db.get_collection("laws")
    laws = collection.find({}, {"description": include_description}).to_list()
    print(laws)


def list_laws():
    db = mongo_client.get_database("pravni-vodnik")
    collection = db.get_collection("laws")
    laws = collection.find(
        {}, {"law_id": 1, "_id": 0, "common_abbreviations": 1}
    ).to_list()

    return laws


def save_document(document: Document, version: int = 1):
    if version == 1:
        col = get_collection("documents")
        result = col.insert_one(document)
        return result.inserted_id
