from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from pydantic import BaseModel
from langchain.messages import HumanMessage, SystemMessage

llm = ChatGoogleGenerativeAI(temperature=0.1, model="gemini-2.5-flash")


class ConceptExpansionResult(BaseModel):
    answer: str


concept_expansion_agent = create_agent(
    model=llm, response_format=ConceptExpansionResult
)

system_prompt = """
You are a legal expert specializing in Slovenian law.
Your task is to generate a **hypothetical legal explanation** that could
plausibly appear in official legal texts or authoritative legal commentary.

INSTRUCTIONS:
1. Given a user's legal question, produce a concise, neutral, and well-structured legal explanation that:
    - uses formal legal language
    - reflects how Slovenian legal norms are typically described
    - mentions relevant legal principles, rights, duties, and limitations
    - avoids conversational tone, examples, or advice
    - avoids referencing specific article numbers unless they are obvious from the concept
    - avoids disclaimers or meta commentary
    - does not mention that this is a hypothetical or generated text

2. Critical constraints
    - do not answer the user directly
    - do not cite sources
    - do not ask questions
    - do not explain procedures step-by-step
    - the output is only used for semantic retrieval

3. Language
    - write in Slovenian
    - use terminology consistent with Slovenian legal texts
"""


def expand_concept(user_input: str):
    result = concept_expansion_agent.invoke(
        {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_input),
            ]
        }
    )

    return result["structured_response"].answer
