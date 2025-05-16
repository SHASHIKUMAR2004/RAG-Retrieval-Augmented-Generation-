# RAG-Retrieval-Augmented-Generation-
Architecture & Flow
📁 1. Document Preprocessing & Embedding
All PDF files or raw documents are first converted into text chunks using a separate script (test_conversion.py or similar).

These chunks are stored in a CSV file with columns like content and source (for traceability).

Using SentenceTransformer (e.g., all-MiniLM-L6-v2), each chunk is embedded into a 384-dimensional vector.

🧠 2. Vector Storage using Qdrant
The vectorized document chunks are upserted into a Qdrant collection (rag_chunks) using the QdrantClient.

Each vector point includes:

A UUID

The embedding vector

A payload containing the text content

Qdrant serves as a high-performance vector database for similarity search.

🧵 3. User Query Processing
When a user enters a query in the Streamlit interface:

The query is embedded using the same SentenceTransformer.

Qdrant performs a vector similarity search to find the top-K most relevant chunks above a similarity threshold.

These context chunks are displayed and used to build a dynamic context.

🤖 4. Response Generation via LLaMA 3 (Groq)
A prompt is constructed combining the retrieved context and the user’s question.

This prompt is passed to the LLaMA 3 model hosted on Groq API, which generates a relevant, context-aware answer.

If no context is found, LLaMA answers based on its general knowledge.

💬 5. Interactive Chat UI (Streamlit)
The frontend mimics ChatGPT-style interaction:

A left sidebar stores search/chat history.

A bottom-aligned text box handles user input.

Responses and queries are styled in a user-agent chat format.

Streamlit handles session management, real-time updates, and smooth interaction flow.

🔄 Summary Workflow
scss
Copy
Edit
PDFs → Text Chunks → Embeddings → Qdrant Vector DB
      ↑                         ↓
   User Query → Embedding → Qdrant Search → Top Chunks → Prompt → LLaMA 3 (Groq) → Response
This modular design ensures scalability, faster semantic search, and high-quality natural language answers using retrieval-augmented generation (RAG).
