from pymongo.mongo_client import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

uri = os.getenv("MONGODB_URI")

mongo_client = MongoClient(uri)


def get_documents_by_ids(doc_ids, db_name="pravni-vodnik", collection_name="articles"):
    db = mongo_client.get_database(db_name)
    collection = db.get_collection(collection_name)
    documents = collection.find({"_id": {"$in": doc_ids}})
    return list(documents)
