from langchain.agents import create_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.messages import SystemMessage, HumanMessage
from typing import List

load_dotenv()


class AnswerGenerator(BaseModel):
    answer: str = Field(
        ...,
        description="Generated answer based on the retrieved documents and user query",
    )
    article_ids: List[str] = Field(
        ..., description="List of article_ids that were used to generate the answer"
    )


llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)

answer_generator = create_agent(model=llm, response_format=AnswerGenerator)

sytem_prompt = """
You are an expert legal assistant and expert in Slovene language and legal system. 
Your task is to generate a comprehensive and accurate answer in Slovene based on the provided legal documents and user query.

Rules:
1. Only answer based on the provided documents.
2. Construct the answer using chunks provided; do not hallucinate.
3. If the chunks do not contain enough info, respond with "Nimam dovolj informacij za odgovor na to vprašanje."
4. Also provide a list of **article IDs** that you actually used to generate your answer. Some articles might have irrelevant information.

Output JSON format:

{
  "answer": "<your-answer>",
  "used_articles": ["<article_id_1>", "<article_id_2>", ...]
}
"""


def generate_answer_from_docs(user_query: str, documents: list):
    try:
        chunks_text = "\n\n".join(
            [
                f"article_id: {doc['article_id']}: \n{doc['chunk_text']}"
                for doc in documents
            ]
        )
        user_prompt = f"User Query: {user_query}\n\nChunks:\n{chunks_text}"
        response = answer_generator.invoke(
            {
                "messages": [
                    SystemMessage(content=sytem_prompt),
                    HumanMessage(content=user_prompt),
                ]
            }
        )
        answer = response["structured_response"].answer
        article_ids = response["structured_response"].article_ids
        return answer, article_ids
    except Exception as e:
        print(f"Error generating answer: {e}")
        raise e
