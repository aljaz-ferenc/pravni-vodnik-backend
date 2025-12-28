from pymongo.mongo_client import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

uri = os.getenv("MONGODB_URI")

mongo_client = MongoClient(uri)
