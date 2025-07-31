import openai
import os
import numpy as np
from .embedder import EMBEDDINGS_INDEX
from dotenv import load_dotenv 

load_dotenv()

openai.api_type = "azure"
openai.api_key = os.getenv("AZURE_OPENAI_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = "2023-05-15"  # Or current

EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")
CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def answer_query(query: str):
    if not EMBEDDINGS_INDEX:
        return {"error": "No embeddings found. Run /embed first."}

    query_embedding = openai.Embedding.create(
        input=query,
        deployment_id=EMBED_DEPLOYMENT
    )["data"][0]["embedding"]

    similarities = [
        (cosine_similarity(query_embedding, doc["embedding"]), doc["text"])
        for doc in EMBEDDINGS_INDEX.values()
    ]

    top_chunks = sorted(similarities, reverse=True)[:3]
    context = "\n\n".join([t[1] for t in top_chunks])

    prompt = f"Answer the question based on the context below:\n\nContext:\n{context}\n\nQuestion: {query}"

    response = openai.ChatCompletion.create(
        deployment_id=CHAT_DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}]
    )

    return {"response": response["choices"][0]["message"]["content"]}
