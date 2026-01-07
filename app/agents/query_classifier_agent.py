from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from app.models.QueryType import QueryType

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)


class QueryClassificationResponse(BaseModel):
    query_type: QueryType = Field(..., description="Query type")


system_prompt = """
You are a legal query classifier agent for Slovenian law. Your task is to classify incoming queries into one of the predefined types. 

Query types:
1. **exact** – The user asks about one or more specific articles or clauses and wants their content explained or summarized (e.g., “What does Article 5 say?” or “What do Articles 5 and 6 say?”).
2. **broad** – The user asks a general topic spanning multiple articles (e.g., “What rights do citizens have regarding education?”).
3. **general** – The user asks about a general legal concept without specifying a law (e.g., “What is the right to property in Slovenia?”).
4. **comparative** – The user asks for a comparison between specific articles (e.g., “Difference between Article 5 and Article 6?”). Only use 'comparative' if specific articles and laws can be extracted from the user's question.
5. **unrelated** - The user query is unrelated to law (e.g., small talk, casual questions, personal topics...)

Provide the classification in the following JSON format (strictly, no extra text):

```json
{
    "query_type": "<exact|broad|general|comparative|unrelated>",
}

"""

agent = create_agent(model=llm, response_format=QueryClassificationResponse)


def classify_query(user_input: str):
    result = agent.invoke(
        {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_input),
            ]
        }
    )

    return result["structured_response"].query_type
