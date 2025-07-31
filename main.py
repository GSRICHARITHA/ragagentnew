from fastapi import FastAPI, UploadFile, File, Form
from utils import file_handler, processor, embedder, responder

app = FastAPI()

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    return await file_handler.save_file(file)

@app.post("/process")
def process_files():
    return processor.process_all()

@app.post("/embed")
def embed_chunks():
    return embedder.generate_embeddings_chroma()

@app.post("/response")
def generate_response(query: str = Form(...)):
    return responder.answer_query(query)
