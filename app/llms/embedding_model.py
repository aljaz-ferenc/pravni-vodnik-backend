from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
from pydantic import SecretStr

load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=1536,
    api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
)
