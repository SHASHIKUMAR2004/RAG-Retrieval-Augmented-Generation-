import streamlit as st
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from llama_index.core import Settings
from llama_index.llms.groq import Groq
from llama_index.core.llms import ChatMessage

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url="YOUR_QDRANT_URL",
    api_key="YOUR_QDRANT_API_KEY"
)

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize LLaMA 3 via Groq
llm = Groq(
    model="llama3-8b-8192",
    api_key="YOUR_GROQ_API_KEY",
    temperature=0.7,
    max_tokens=512,
    default_headers={},
    logprobs=None         
)

# Set the LLM globally
Settings.llm = llm

# Function to embed query
def embed_query(query: str):
    return embed_model.encode([query])[0]

# Function to search Qdrant
def search_qdrant(query_vector, threshold=0.35, top_k=5):
    search_results = qdrant_client.search(
        collection_name="rag_chunks",
        query_vector=query_vector,
        limit=top_k,
    )
    return [r for r in search_results if r.score >= threshold]

# Streamlit UI
st.title("🔎 RAG System with Qdrant and LLaMA 3 (Groq API)")

user_query = st.text_input("Enter your query:")
if user_query:
    st.write("Searching for relevant documents in Qdrant...")

    query_vector = embed_query(user_query)
    results = search_qdrant(query_vector)

    st.write("### Top results from Qdrant:")
    context_chunks = []
    for idx, r in enumerate(results, 1):
        chunk = r.payload.get("content", "")
        context_chunks.append(chunk)
        st.markdown(f"*Result {idx} (Score: {round(r.score, 4)}):*\n\n{chunk}\n")

    # Combine chunks for LLaMA input
    combined_context = "\n".join(context_chunks).strip()

    st.write("Generating response from LLaMA 3 (Groq)...")

    # Build the prompt
    if combined_context:
        prompt = f"""Answer the following question using the context provided below. If the context is not relevant, answer using your own knowledge.

Context:
{combined_context}

Question: {user_query}
Answer:"""
    else:
        prompt = f"Answer the following question using your own knowledge.\n\nQuestion: {user_query}\nAnswer:"

    # Generate response
    try:
        response = llm.complete(prompt=prompt)
        st.markdown("### Response from LLaMA 3:\n" + response.text)
    except Exception as e:
        st.error(f"Error generating response: {e}")
