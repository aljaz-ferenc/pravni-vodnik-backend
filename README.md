# Pravni Vodnik

Pravni Vodnik allows you to ask questions about Slovenian law in natural language, ranging from specific legal articles to broader topics. The application retrieves relevant legal sources, synthesizes structured explanations, and generates documents that cite the underlying legislation. Each generated document is saved, allowing users to review and revisit their queries.

> ⚠️ **Important**: Pravni Vodnik is **not** a substitute for professional legal advice. The generated content is informational only.

## Tech Stack

### Backend

* Python
* LangGraph
* MongoDB (article storage)
* Pinecone (vector store)
* Server-Sent Events (SSE)

### Frontend

* Next.js

## Architecture Overview

Pravni Vodnik is built as a **Retrieval-Augmented Generation (RAG) system**. The main components are:

### Data Ingestion
- Legal texts are split into **chunks**, typically individual paragraphs of an article.  
- Each chunk is converted into a **vector embedding** using OpenAI’s embedding models.  
- Chunks are stored in **Pinecone** for fast semantic search, while full articles are stored in **MongoDB**.  

### Query Handling
1. **Query Classification** – Determines whether the user question is an **exact article reference**, a **broad topic**, a **general concept**, or **unrelated to law**.  
2. **Routing** – Depending on the classification, the query is routed to the appropriate retrieval pipeline.  
  

### Retrieval Strategies
- **Exact queries**: Direct lookup of the article in MongoDB.  
- **Broad queries**: Generate multiple sub-queries, retrieve relevant chunks from Pinecone, rerank, and fetch full articles from MongoDB.  
- **General queries**: Expand the query conceptually (HyDE-style), retrieve candidate chunks, rerank, and fetch corresponding articles.  
 - **Unrelated queries** are detected early and gracefully exited, preventing unnecessary processing.  
- **Handling unsupported laws**: If a user asks about a law not currently supported, reranking and score thresholds allow the system to detect low-confidence results and safely indicate that the answer does not exist in the supported corpus.  

### Synthesis & Document Generation
- Retrieved articles and chunks are passed to the **synthesizer agent**, which generates a coherent, structured Markdown document.  
- Each document is saved for later access, allowing users to **review previous queries and generated explanations**.  

### Real-Time Updates
- The system uses **Server-Sent Events (SSE)** to update users on progress at each stage: classification, retrieval, reranking, and document synthesis.


## Supported Laws

The system currently supports a **limited subset** of Slovenian legislation, such as:

* Ustava Republike Slovenije
* Kazenski zakonik (KZ-1)
* Zakon o kazenskem postopku (ZKP)

## Legal Disclaimer

The application provides automatically generated legal information based on available legal texts. The content:

* does **not** constitute legal advice
* may be incomplete or outdated
* should not be relied upon in legal proceedings

Users should always consult a qualified legal professional for authoritative advice.

## Project Status

Pravni Vodnik is fully functional for exploring supported Slovenian laws. Users can ask questions, generate structured legal documents, and access previously generated documents. 

Planned enhancements include:
- Expanding the legal corpus to cover additional Slovenian legislation.
- Implementing a hybrid search that combines semantic and lexical approaches for more accurate retrieval.
- Adding **document versioning**, allowing users to ask follow-up questions or request reformatting to create and store multiple versions of the same document.



