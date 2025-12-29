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
The answer should be in marked down format and written in Slovene language.
Use the provided documents to construct your answer, ensuring that it is relevant to the user's query.
If the documents do not contain sufficient information to answer the query, respond with "Nimam dovolj informacij za odgovor na to vprašanje."
Format the output as a JSON object with a single key "answer" containing the generated answer string.
Example Output:
{
  "answer": "Državni zbor je najvišji organ državne oblasti v Sloveniji in je sestavljen iz 90 poslancev, ki jih volijo državljani na neposrednih volitvah za obdobje štirih let. Njegove glavne naloge vključujejo sprejemanje zakonov, potrjevanje državnega proračuna, nadzor nad delom vlade in zastopanje države v mednarodnih zadevah."
}
"""


def generate_answer(user_query: str, documents: list[dict]) -> AnswerGenerator:
    try:
        docs_content = "\n\n".join(
            [f"Document {i + 1}:\n{doc['text']}" for i, doc in enumerate(documents)]
        )
        user_prompt = f"User Query: {user_query}\n\nDocuments:\n{docs_content}"
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
