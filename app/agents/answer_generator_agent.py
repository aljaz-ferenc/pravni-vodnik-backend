from langchain.agents import create_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.messages import SystemMessage, HumanMessage

load_dotenv()


class AnswerGenerator(BaseModel):
    answer: str = Field(
        ...,
        description="Generated answer based on the retrieved documents and user query",
    )


llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)

answer_generator = create_agent(model=llm, response_format=AnswerGenerator)

sytem_prompt = """
You are an expert legal assistant and expert in slovene language and legal system. Your task is to generate a comprehensive and accurate answer in Slovene language based on the provided legal documents and user query.
The answer should be written in Slovene language.
Only answer based on provided documents.
Use the provided chunks to construct your answer, ensuring that it is relevant to the user's query.
If the chunks do not contain sufficient information to answer the query, respond with "Nimam dovolj informacij za odgovor na to vprašanje."
Format the output as a JSON object with a single key "answer" containing the generated answer string.
Example Output:
{
  "answer": "<your-answer>"
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
        return response["structured_response"].answer
    except Exception as e:
        print(f"Error generating answer: {e}")
        raise e
