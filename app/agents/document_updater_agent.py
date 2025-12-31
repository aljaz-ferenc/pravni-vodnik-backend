from langchain.agents import create_agent
from langchain.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field


class DocumentUpdate(BaseModel):
    document: str = Field(
        ...,
        description="Updated document created by combining the old document with new answer.",
    )


document_updater_agent = create_agent(
    model="gpt-4.1-mini", response_format=DocumentUpdate
)


system_prompt = """
    You are a Slovenian law legal assistant.
    Your are provided with a document from earlier agents.
    You are also provided an answer from the user's latest question and the question itself.
    Your job is to combine the answer and the document into one document and
    return the updated document that contains previous and new information.
    Return the document in markdown format. Use headings, lists, paragraphs and any other markdown feature you think is appropriate.
    You can update the text, add new sections etc.
    Use professional language in Slovene.
"""


def update_document(old_doc: str, new_answer: str):
    result = document_updater_agent.invoke(
        {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=f"""
            PREVIOUS DOCUMENT: {old_doc}\n\n
            NEW ANSWER: {new_answer}
        """
                ),
            ]
        }
    )

    return result["structured_response"].document
