"""RAG module: FAISS vector store with sentence-transformer embeddings."""

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from knowledge_base import GRID_KNOWLEDGE


@st.cache_resource
def build_vector_store():
    """Embed grid knowledge chunks and build in-memory FAISS index."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_texts(GRID_KNOWLEDGE, embeddings)
    return vector_store


def retrieve(query: str, k: int = 4) -> list[str]:
    """Similarity search over grid knowledge, return top-k text chunks."""
    vs = build_vector_store()
    docs = vs.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]
