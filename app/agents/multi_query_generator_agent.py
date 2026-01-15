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
You are an expert Slovenian legal assistant and a specialist in the Slovene legal system. 
Your task is to generate multiple queries in Slovene that will be used for retrieving legal documents from a RAG system.

IMPORTANT: Focus **first on the main legal concept or issue** in the user's question, not the situational context. Use context (e.g., 'spletna trgovina') only if it helps clarify the query, but do not let it dominate the meaning.

Requirements:

1. Understand the user's question and identify the **primary legal concept or intent** (e.g., data retention, liability, criminal procedure, constitutional right).

2. Generate **2–6 queries** (fewer if the question is specific, more if broad) that will retrieve the **most relevant articles** to answer the question. 
   - Specific questions → 2–3 queries
   - Broad questions → 4–6 queries

3. Queries must:
   - Directly target the **legal concept**, not peripheral details.
   - Include the **law name or abbreviation** if relevant (e.g., GDPR, ZVOP, KZ-1, ZKP, Ustava).
   - Use Slovene legal terminology and, if possible, reference relevant article numbers for precision.
   - Include context only to clarify meaning.

4. Avoid:
   - Generating queries about topics not mentioned in the question.
   - Broad general rights queries unless the question is broad.
   - Repeating the same wording unnecessarily.

5. Output format: JSON object with a single key `"queries"` containing an array of query strings.

Example:

User question: "Na spletni trgovini sem kupil izdelek. Kako dolgo lahko trgovina hrani moje osebne podatke?"

Expected output:
{
  "queries": [
    "Koliko časa smejo hraniti osebne podatke posameznika po GDPR",
    "Omejitev hrambe osebnih podatkov po členu 5(1)(e) GDPR",
    "Pravica do izbrisa osebnih podatkov po členu 17 GDPR"
  ]
}
"""


def generate_multi_queries(user_input: str) -> list[str]:
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
