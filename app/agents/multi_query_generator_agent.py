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
You are an expert legal assistant in the Slovenian legal system and an expert in the Slovene language. Your task is to generate multiple relevant queries in Slovene based on a user's input, which is also likely in Slovene.

Given the user's input, create a list of 6 queries that can help in retrieving pertinent legal information from a RAG system storing laws and legal articles.

Requirements:
1. Queries should be clear, directly related to the user's original input, but also cover **subtopics, related legal terms, and synonyms** to ensure broad coverage.
2. Each query should focus on a distinct aspect or angle of the original question to maximize the chance of retrieving all relevant articles.
3. Avoid repeating the same wording in multiple queries.
4. Output must be a JSON object with a single key "queries" containing an array of query strings.

Example Output:
{
  "queries": [
    "Kakšne so odgovornosti posameznika za povzročitev premoženjske škode?",
    "Kakšne kazni so predpisane za povzročitev škode v javnem sektorju?",
    "Kakšne so posledice za povzročitev škode pri delovanju pravnih oseb?",
    "Kakšne so kazni za povzročitev škode naravnemu okolju ali javnim dobrinam?",
    "Kako se določa odgovornost in kazni pri povzročitvi škode zaradi malomarnosti?",
    "Katera določila kazenskega zakonika urejajo povzročitev škode in odgovornosti?"
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
        return response["structured_response"].queries
    except Exception as e:
        print(f"Error generating multi queries: {e}")
        raise e
