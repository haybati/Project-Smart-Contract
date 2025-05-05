import pytesseract
import re
from pdf2image import convert_from_path
import nltk
import tiktoken
import os
import pickle
import faiss
import numpy as np
from openai import OpenAI
from nltk.tokenize.punkt import PunktSentenceTokenizer

# 🔐 OpenAI API key
client = OpenAI(api_key="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")  # Replace with your actual key

# Path setup
nltk_data_path = r"C:\Users\Salman Haybati\AppData\Local\Programs\Python\Python313\nltk_data"
punkt_path = os.path.join(nltk_data_path, "tokenizers", "punkt", "english.pickle")
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
nltk.data.path.append(nltk_data_path)

# Load sentence tokenizer manually
with open(punkt_path, "rb") as f:
    sentence_tokenizer = pickle.load(f)

def sent_tokenize(text):
    return sentence_tokenizer.tokenize(text)

encoding = tiktoken.encoding_for_model("text-embedding-ada-002")

def clean_ocr_text(page_text):
    page_text = re.sub(r'Page\s*\d+|^\d+\s*/\s*\d+', '', page_text, flags=re.IGNORECASE)
    page_text = re.sub(r'FIDIC.*Contract.*', '', page_text, flags=re.IGNORECASE)
    page_text = re.sub(r'\s{2,}', ' ', page_text)
    page_text = re.sub(r'\n{2,}', '\n', page_text)
    page_text = re.sub(r'[^\x00-\x7F]+', ' ', page_text)
    return page_text.strip()

def ocr_pdf_to_text(pdf_path):
    images = convert_from_path(pdf_path, dpi=300)
    text = ""
    for i, img in enumerate(images):
        try:
            try:
                osd = pytesseract.image_to_osd(img)
                rotation = int(re.search("Rotate: (\d+)", osd).group(1))
                if rotation != 0:
                    img = img.rotate(rotation, expand=True)
            except Exception as e:
                print(f"⚠️ Skipping OSD for page {i+1}: {e}")

            raw = pytesseract.image_to_string(img, lang='eng', config="--psm 6")
            cleaned = clean_ocr_text(raw)
            text += f"\n\n--- Page {i+1} ---\n\n{cleaned}"
        except Exception as e:
            print(f"❌ Error processing page {i+1}: {e}")
    return text

def sentence_chunking(text, max_tokens=1000, overlap_sentences=1):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_tokens = 0
    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        sentence_tokens = len(encoding.encode(sentence))
        if current_tokens + sentence_tokens > max_tokens:
            chunks.append(" ".join(current_chunk))
            i = max(0, i - overlap_sentences)
            current_chunk = []
            current_tokens = 0
        else:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
            i += 1
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def get_embedding(text, model="text-embedding-ada-002"):
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def save_faiss_index(chunks, index_dir="faiss_index"):
    print("🔄 Embedding chunks and building FAISS index...")
    embeddings = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        try:
            emb = get_embedding(chunk)
            embeddings.append(emb)
            metadatas.append({"chunk_id": i, "text": chunk})
        except Exception as e:
            print(f"❌ Error embedding chunk {i}: {e}")

    if not embeddings:
        print("❌ No embeddings created. Aborting FAISS save.")
        return

    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(index_dir, "index.faiss"))
    with open(os.path.join(index_dir, "index.pkl"), "wb") as f:
        pickle.dump(metadatas, f)

    print(f"✅ FAISS index saved to '{index_dir}/index.faiss'")
    print(f"✅ Metadata saved to '{index_dir}/index.pkl'")

def process_pdf(pdf_path, output_txt_file, metadata_file, chunk_size=1000, overlap_sentences=1):
    print("📄 Converting PDF to text...")
    cleaned_text = ocr_pdf_to_text(pdf_path)
    with open(output_txt_file, "w", encoding="utf-8") as f:
        f.write(cleaned_text)
    print(f"✅ Cleaned text saved to {output_txt_file}")

    print("✂️ Chunking text by sentence with overlap...")
    chunks = sentence_chunking(cleaned_text, chunk_size, overlap_sentences)
    with open(metadata_file, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"Chunk {i+1}:\n{chunk}\n\n--- End of Chunk {i+1} ---\n\n")
    print(f"✅ Metadata saved to {metadata_file}")
    print(f"✅ Total chunks: {len(chunks)}")

    save_faiss_index(chunks, index_dir="faiss_index")

if __name__ == "__main__":
    process_pdf(
        pdf_path=r"C:\Users\Salman Haybati\Documents\1. SALMAN\10. PLN\Prajab\2. OJT\1. Laporan dan Penugasan\8. PA\1. Smart Contract V2\contracts_library\Kontrak Original PLN_SULUT BUKU 1a.pdf",
        output_txt_file="cleaned_text.txt",
        metadata_file="chunks_metadata.txt",
        chunk_size=1000,
        overlap_sentences=1
    )
