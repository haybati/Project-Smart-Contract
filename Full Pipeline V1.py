import os
import fitz  # PyMuPDF
import openai
import json
import tiktoken
from tqdm import tqdm
from PIL import Image

# Set your API key
openai.api_key = "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"

# Set your folder path
PDF_FOLDER = r"C:\\Users\\Salman Haybati\\Documents\\1. SALMAN\\10. PLN\\Prajab\\2. OJT\\1. Laporan dan Penugasan\\8. PA\\1. Smart Contract V2\\contracts_library"
OUTPUT_TEXT_FILE = os.path.join(PDF_FOLDER, "output_text.txt")
CHUNK_FILE = os.path.join(PDF_FOLDER, "chunk.txt")
EMBEDDING_FILE = os.path.join(PDF_FOLDER, "embedding.jsonl")

# Helper to convert landscape to portrait
def correct_orientation(page):
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    if pix.width > pix.height:
        img = img.rotate(270, expand=True)
    return img

# Step 1 & 2: Scan PDFs and extract text
all_text = []
for filename in os.listdir(PDF_FOLDER):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(PDF_FOLDER, filename)
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            image = correct_orientation(page)
            text += page.get_text()
        all_text.append(text)

# Save extracted full text
with open(OUTPUT_TEXT_FILE, "w", encoding="utf-8") as f:
    f.write("\n\n".join(all_text))

# Step 3: Chunking paragraf max 2000 token

def split_chunks_paragraph(text, max_tokens=2000):
    encoding = tiktoken.encoding_for_model("text-embedding-3-small")
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        tokens = encoding.encode(para)
        current_tokens = encoding.encode(current_chunk)

        if len(tokens) > max_tokens:
            sentences = para.split('. ')
            temp = ""
            for sentence in sentences:
                if len(encoding.encode(temp + sentence + ". ")) < max_tokens:
                    temp += sentence + ". "
                else:
                    if temp: chunks.append(temp.strip())
                    temp = sentence + ". "
            if temp:
                chunks.append(temp.strip())
        elif len(current_tokens) + len(tokens) < max_tokens:
            current_chunk += "\n" + para
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# Load full text
with open(OUTPUT_TEXT_FILE, "r", encoding="utf-8") as f:
    full_text = f.read()

chunks = split_chunks_paragraph(full_text)

# Save chunk to file
with open(CHUNK_FILE, "w", encoding="utf-8") as f:
    for chunk in chunks:
        f.write(chunk + "\n\n")

# Step 4: Embedding with ada-002 (text-embedding-3-small)
def get_embedding(text):
    response = openai.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

with open(EMBEDDING_FILE, "w", encoding="utf-8") as f:
    for chunk in tqdm(chunks):
        embedding = get_embedding(chunk)
        record = {"text": chunk, "embedding": embedding}
        f.write(json.dumps(record) + "\n")

print("Pipeline selesai. Semua output disimpan di folder:", PDF_FOLDER)
