from langchain.agents import create_agent
from langchain.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field


class DocumentUpdate(BaseModel):
    document: str = Field(
        ...,
        description="Updated document created by combining the old document with new answer.",
    )
    title: str = Field(description="Title of the document")


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
    You can also update the title so it matches the added content.
    Do NOT include the title in the 'document'. Put the title in the 'title' field.
"""


def synthesize_document(old_doc: str, new_answer: str, user_input: str, title: str):
    result = document_updater_agent.invoke(
        {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=f"""
            USER'S QUESTION: {user_input}\n\n
            PREVIOUS DOCUMENT: {title}\n {old_doc}\n\n
            NEW ANSWER: {new_answer}
        """
                ),
            ]
        }
    )

    return result["structured_response"].document, result["structured_response"].title
