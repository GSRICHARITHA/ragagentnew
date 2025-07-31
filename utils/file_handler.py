import os
import shutil
from fastapi import UploadFile

UPLOAD_DIR = "uploaded_files/"

async def save_file(file: UploadFile):
    if not file.filename.endswith(('.pdf', '.docx')):
        return {"error": "Only .pdf and .docx files are supported."}
    
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"filename": file.filename, "status": "uploaded"}
