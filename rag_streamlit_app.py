import streamlit as st
import openai
import json
import numpy as np
import tiktoken
from sklearn.metrics.pairwise import cosine_similarity

# Konfigurasi awal
openai.api_key = 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  # Ganti dengan API Key Anda
EMBEDDING_FILE = "embedding.jsonl"
MODEL = "text-embedding-3-small"

# Load semua embedding dari file
@st.cache_data
def load_embeddings(filepath):
    chunks = []
    vectors = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            chunks.append(obj["text"])
            vectors.append(obj["embedding"])
    return chunks, np.array(vectors)

# Embed pertanyaan
def get_query_embedding(query):
    response = openai.embeddings.create(
        input=[query],
        model=MODEL
    )
    return response.data[0].embedding

# Cari top-n chunk paling relevan
def search_chunks(query, chunks, vectors, top_k=3):
    query_vector = np.array(get_query_embedding(query)).reshape(1, -1)
    sims = cosine_similarity(vectors, query_vector).flatten()
    top_indices = sims.argsort()[-top_k:][::-1]
    results = [(chunks[i], sims[i]) for i in top_indices]
    return results

# App Streamlit
st.set_page_config(page_title="Smart Contract RAG", layout="wide")
st.title("🔍 Smart Contract Retrieval (RAG)")

query = st.text_input("Masukkan pertanyaan Anda:")

if query:
    with st.spinner("Mencari jawaban..."):
        chunks, vectors = load_embeddings(EMBEDDING_FILE)
        top_chunks = search_chunks(query, chunks, vectors, top_k=3)
        
        st.subheader("📄 Konteks yang ditemukan:")
        for i, (chunk, score) in enumerate(top_chunks, 1):
            st.markdown(f"**Chunk {i} (Score: {score:.4f})**")
            st.markdown(f"```\n{chunk}\n```")
        
        # Opsional: kirim ke ChatGPT untuk menjawab berdasarkan context
        context = "\n\n".join([c for c, _ in top_chunks])
        messages = [
            {"role": "system", "content": "Anda adalah asisten ahli kontrak. Jawablah pertanyaan dengan berdasarkan konteks yang diberikan."},
            {"role": "user", "content": f"Pertanyaan: {query}\n\nKonteks:\n{context}"}
        ]
        completion = openai.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        answer = completion.choices[0].message.content
        st.subheader("💬 Jawaban:")
        st.write(answer)
