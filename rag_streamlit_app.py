import streamlit as st
import openai
import json
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Set API key dari secret
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load embeddings
@st.cache_resource
def load_embeddings(filepath):
    chunks = []
    embeddings = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            chunks.append(item["text"])
            embeddings.append(item["embedding"])
    return chunks, np.array(embeddings)

# Hitung similarity dan ambil top-N
def get_top_chunks(query_embedding, embeddings, chunks, top_k=10):
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

# Kirim query ke OpenAI
def query_rag(query, chunks, embeddings):
    # Aturan yang wajib ditambahkan
    rules = (
        "Berikut adalah aturan yang WAJIB diikuti dalam menjawab pertanyaan:\n"
        "1. Sertakan nomor klausul secara lengkap.\n"
        "2. Sertakan kutipan potongan kalimat asli dari klausul.\n"
        "3. Tidak boleh mengada-ada atau menyimpulkan di luar isi kontrak.\n\n"
    )

    # Embed query
    response = openai.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    query_embedding = response.data[0].embedding

    # Ambil top context
    top_chunks = get_top_chunks(query_embedding, embeddings, chunks)
    context = "\n\n".join(top_chunks)

    # Buat prompt lengkap
    prompt = rules + f"Jawablah pertanyaan berikut berdasarkan konteks:\n\n{context}\n\nPertanyaan: {query}\n\nJawaban:"

    # Kirim ke OpenAI chat
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content

# Streamlit UI
st.set_page_config(page_title="RAG Smart Contract App", layout="wide")
st.title("📄🔍 RAG: Smart Contract QA")

filepath = "embedding.jsonl"  # Pastikan file ini ada di repo dan bukan expired

if not os.path.exists(filepath):
    st.error("❌ File embedding.jsonl tidak ditemukan. Upload ulang atau periksa nama file.")
else:
    chunks, embeddings = load_embeddings(filepath)
    query = st.text_area("Masukkan pertanyaan terkait kontrak:", height=150)

    if st.button("Jalankan RAG"):
        if query.strip() == "":
            st.warning("⚠️ Pertanyaan tidak boleh kosong.")
        else:
            with st.spinner("Menjalankan model..."):
                try:
                    answer = query_rag(query, chunks, embeddings)
                    st.success("✅ Jawaban ditemukan:")
                    st.markdown(answer)
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memproses: {e}")
