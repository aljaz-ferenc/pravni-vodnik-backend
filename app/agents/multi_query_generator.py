from langchain.agents import create_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.messages import SystemMessage, HumanMessage

load_dotenv()


class MultiQueryGenerator(BaseModel):
    queries: list[str] = Field(
        ...,
        description="List of generated queries based on the users input in Slovene language",
    )


llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)

multi_query_generator = create_agent(model=llm, response_format=MultiQueryGenerator)

system_prompt = """
You are an expert legal assistant and expert in slovene language and legal system. Your task is to generate multiple relevant queries in Slovene language based on a user's input, which is likely to also be in Slovene language.
Given the user's input, create a list of three queries that can help in retrieving pertinent legal information from a RAG system.
Ensure that the queries are clear and directly related to the user's original input but also broad enough to retrieve all information needed. Information is stored as a collection of articles.
Format the output as a JSON object with a single key "queries" containing an array of query strings.
Example Output:
{
  "queries": [
    "Kdo ima oblast v Sloveniji?",
    "Kateri so glavni pravni viri v Sloveniji?",
    "Kako je organizirana sodna veja oblasti v Sloveniji?"
  ]
}
"""


def generate_multi_queries(user_input: str) -> MultiQueryGenerator:
    try:
        response = multi_query_generator.invoke(
            {
                "messages": [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_input),
                ]
            }
        )
        return response["structured_response"]
    except Exception as e:
        print(f"Error generating multi queries: {e}")
        raise e
