import os
import fitz  # PyMuPDF
import docx
from pathlib import Path

UPLOAD_DIR = "uploaded_files/"
CHUNK_DIR = "processed_chunks/"
CHUNK_SIZE = 500  # words

def read_docx(path):
    doc = docx.Document(path)
    return " ".join([p.text for p in doc.paragraphs])

def read_pdf(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=100):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap  # slide window
    return chunks
def process_all():
    os.makedirs(CHUNK_DIR, exist_ok=True)
    count = 0

    for file in Path(UPLOAD_DIR).glob("*"):
        text = ""
        if file.suffix == ".docx":
            text = read_docx(file)
        elif file.suffix == ".pdf":
            text = read_pdf(file)

        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            chunk_path = Path(CHUNK_DIR) / f"{file.stem}_chunk_{i}.txt"
            with open(chunk_path, "w", encoding="utf-8") as f:
                f.write(chunk)
            count += 1

    return {"status": "processed", "chunks_created": count}
