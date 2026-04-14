import os
import shutil
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from rag.indexing import create_vector_store
from app.pipeline import RAGPipeline
from app.config import settings

app = FastAPI()

UPLOAD_DIR = "data"
VECTOR_DIR = settings.VECTOR_DB_PATH

pipeline = None

class QueryRequest(BaseModel):
    query: str


@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    global pipeline

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    create_vector_store(
        file_path=file_path,
        save_path=VECTOR_DIR
    )

    pipeline = RAGPipeline(store_path=VECTOR_DIR)

    return {"message": f"{file.filename} uploaded and indexed successfully"}

@app.post("/ask")
def ask(request: QueryRequest):
    global pipeline

    if pipeline is None:
        return {"error": "No document uploaded yet"}

    response = pipeline.run(request.query)
    return {"answer": response}


@app.get("/")
def home():
    return {"message": "RAG API running"}