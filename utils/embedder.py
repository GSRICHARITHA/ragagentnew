import json
import os
import openai
from pathlib import Path
from dotenv import load_dotenv 

load_dotenv()

openai.api_type = "azure"
openai.api_key = os.getenv("AZURE_OPENAI_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = "2023-05-15" 

DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")
CHUNK_DIR = "processed_chunks/"
EMBEDDINGS_FILE = "embeddings_index.json"
EMBEDDINGS_INDEX = {}

def generate_embeddings():
    try:
        if not os.path.isdir(CHUNK_DIR):
            return {"error": f"{CHUNK_DIR} folder not found."}

        files = os.listdir(CHUNK_DIR)
        if not files:
            return {"error": "No chunk files found. Run /process first."}

        for filename in files:
            file_path = os.path.join(CHUNK_DIR, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            if not text.strip():
                continue

            response = openai.Embedding.create(
                input=text,
                deployment_id=DEPLOYMENT
            )

            embedding = response["data"][0]["embedding"]

            EMBEDDINGS_INDEX[filename] = {
                "embedding": embedding,
                "text": text
            }

        # ✅ Save to file
        with open(EMBEDDINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(EMBEDDINGS_INDEX, f)

        return {"status": "embeddings generated and saved", "documents": len(EMBEDDINGS_INDEX)}

    except Exception as e:
        return {"error": str(e)}

def load_embeddings():
    global EMBEDDINGS_INDEX
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
            EMBEDDINGS_INDEX = json.load(f)