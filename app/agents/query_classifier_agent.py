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
    reasoning: str = Field(..., description="Your reasoning")


system_prompt = """
You are a legal query classifier agent for Slovenian law.

Your task is to classify incoming user queries into one of the predefined types.
The query does NOT need to explicitly mention a law or article to be considered legal.

IMPORTANT:
Questions about legal responsibility, liability, criminal acts, duties, rights, sanctions,
or legal consequences (even when written informally or in the first person)
ARE considered legal questions.

Query types:

1. **exact**
– The user asks about one or more specific articles or clauses and wants their content explained or summarized
  (e.g., “What does Article 5 say?” or “What do Articles 5 and 6 say?”).

2. **broad**
– The user asks about a legal topic that likely spans multiple articles within one or more laws
  (e.g., “What rights do citizens have regarding education?”).

3. **general**
– The user asks about a general legal concept, legal responsibility, or legal consequences
  without specifying a particular law or article
  (e.g., “What is criminal liability?”, “What are the legal consequences if someone commits a crime?”).

4. **unrelated**
– The user query is clearly unrelated to law
  (e.g., small talk, casual conversation, technical questions unrelated to law, personal opinions).

Provide the classification in the following JSON format (strictly, no extra text):

```json
{
  "query_type": "<exact|broad|general|unrelated>"
  "reasoning": "Short explanation (1–2 sentences) explaining why this query was classified this way."
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
