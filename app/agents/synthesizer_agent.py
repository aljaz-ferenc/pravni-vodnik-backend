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
You are provided with:
- a document from earlier questions (if available),
- the answer from the user's latest question, and
- the user's latest question itself.

Your job is to combine the new answer with the previous document (if there is one) into a single coherent document. 
- Return the document in markdown format. Use subheadings, lists, paragraphs, and any other markdown features as appropriate.
- You can update the text, add new sections, etc.
- Use professional Slovene language.
- If there is no previous document, just format the new answer as the document.
- IMPORTANT: Do NOT include the title (# header) in the markdown. Put the title in the 'title' field. You may optionally update the title so it matches the content.
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
