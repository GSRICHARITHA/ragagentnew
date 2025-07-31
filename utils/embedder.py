import os
from pathlib import Path
from dotenv import load_dotenv
import openai
import chromadb
from chromadb.config import Settings

# Load environment variables
load_dotenv()

# Configure Azure OpenAI
openai.api_type = "azure"
openai.api_key = os.getenv("AZURE_OPENAI_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = "2023-05-15"

DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")
CHUNK_DIR = "processed_chunks/"

# Set up Chroma client and collection
client = chromadb.Client(Settings(anonymized_telemetry=False))
collection = client.get_or_create_collection("rag_chunks")

def generate_embeddings_chroma():
    try:
        if not os.path.isdir(CHUNK_DIR):
            return {"error": f"{CHUNK_DIR} folder not found."}

        files = os.listdir(CHUNK_DIR)
        if not files:
            return {"error": "No chunk files found. Run /process first."}

        count = 0
        for filename in files:
            file_path = os.path.join(CHUNK_DIR, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip()

            if not text:
                continue

            response = openai.Embedding.create(
                input=text,
                deployment_id=DEPLOYMENT
            )
            embedding = response["data"][0]["embedding"]

            # Add to Chroma collection
            collection.add(
                documents=[text],
                embeddings=[embedding],
                ids=[filename]
            )

            count += 1

        return {"status": "embeddings generated and stored in ChromaDB", "documents": count}

    except Exception as e:
        return {"error": str(e)}
