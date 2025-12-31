from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool
from pydantic import BaseModel, Field
from app.database.mongo import mongo_client
from app.database.mongo import list_laws
from langchain.messages import HumanMessage, SystemMessage

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1)


class ExactArticleResponse(BaseModel):
    answer: str = Field(description="Answer to the user's question")
    article_ids: list[str] = Field(
        description="List of article ids the user asked about"
    )


@tool(
    "get_article",
    description="Get specific article based on law_id and article_number. Use this for fetching a specific law article.",
)
def get_article(law_id: str, article_number: str):
    """
    Get specific article based on law_id and article_number

    args:
        - law_id: str (e.g., 'ustava')
        - article_number: str ('3', '3.a')
    """
    db = mongo_client.get_database("pravni-vodnik")
    col = db.get_collection("articles")
    article = col.find_one({"law_id": law_id, "article_number": article_number})
    return article


exact_article_agent = create_agent(
    model="gpt-4.1-mini", response_format=ExactArticleResponse, tools=[get_article]
)

system_prompt = f"""
You are a legal assistant. Your job is to answer the user's question by fetching the specific articles from the database.
Instructions:
1. Return the answer in markdown format.
2. Identify the law_id and article_number(s) from the user's question.
3. For each article_number, call the tool `get_article` with law_id and article_number.
4. If an article is not found, indicate that in your answer.
5. Combine the contents of all fetched articles to answer the question.
6. Return the result in JSON format with:
    - answer: str
    - article_ids: list of article ids you fetched.
7. IMPORTANT: In your response, the field 'article_ids' must contain the **'_id'** from the fetched article(s), NOT the article_number.
8. Be concise and do not stray too far from the article text.

Available laws: {list_laws()}
"""


def run_exact_article_agent(user_input):
    result = exact_article_agent.invoke(
        {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_input),
            ]
        }
    )

    return result["structured_response"].answer, result[
        "structured_response"
    ].article_ids
