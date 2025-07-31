import os
import openai
import numpy as np
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from chromadb import PersistentClient

load_dotenv()

# Azure OpenAI config
openai.api_type = "azure"
openai.api_key = os.getenv("AZURE_OPENAI_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = "2023-05-15"

EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")
CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

# ✅ Persistent ChromaDB path (adjust for local or Azure)
CHROMA_PATH = "/home/site/data/chroma_db"  # for Azure

# ✅ Ensure chroma path exists
os.makedirs(CHROMA_PATH, exist_ok=True)

# ✅ Set up persistent Chroma client
client = PersistentClient(path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))
collection = client.get_or_create_collection("rag_chunks")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def answer_query(query: str):
    try:
        # Step 1: Generate embedding for the query
        response = openai.Embedding.create(
            input=query,
            deployment_id=EMBED_DEPLOYMENT
        )
        query_embedding = response["data"][0]["embedding"]

        # Step 2: Search top 3 similar documents from ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )

        documents = results["documents"][0] if results["documents"] else []

        if not documents:
            return {"response": "No relevant documents found."}

        # Step 3: Compose prompt
        context = "\n\n".join(documents)
        prompt = f"Answer the question based on the context below:\n\nContext:\n{context}\n\nQuestion: {query}"

        # Step 4: Query Azure OpenAI
        chat_response = openai.ChatCompletion.create(
            deployment_id=CHAT_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}]
        )

        return {
            "response": chat_response["choices"][0]["message"]["content"]
        }

    except Exception as e:
        return {"error": str(e)}
