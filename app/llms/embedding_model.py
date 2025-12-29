from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", dimensions=1536, api_key=os.getenv("OPENAI_API_KEY")
)
