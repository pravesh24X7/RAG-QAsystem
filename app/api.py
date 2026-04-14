import os
import shutil
import time
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
from rag.indexing import create_vector_store
from app.pipeline import RAGPipeline
from slowapi import Limiter
from slowapi.util import get_remote_address


app = FastAPI()

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

UPLOAD_DIR = "data"
VECTOR_BASE_DIR = "vector_store"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_BASE_DIR, exist_ok=True)

pipelines = {}


class QueryRequest(BaseModel):
    query: str
    document_id: str


def process_document(file_path: str, vector_path: str):
    create_vector_store(file_path=file_path, save_path=vector_path)


@app.post("/upload")
def upload_file(
    file: UploadFile,
    background_tasks: BackgroundTasks = None
):

    if not file.filename.endswith((".pdf", ".txt")):
        return {"error": "Only PDF and TXT files are supported"}

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    vector_path = os.path.join(VECTOR_BASE_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    background_tasks.add_task(process_document, file_path, vector_path)

    return {
        "message": "File uploaded successfully. Processing in background.",
        "document_id": file.filename
    }


@app.post("/ask")
@limiter.limit("5/minute")
def ask(request: QueryRequest):

    vector_path = os.path.join(VECTOR_BASE_DIR, request.document_id)

    if not os.path.exists(vector_path):
        return {"error": "Document not processed yet. Try again later."}

    if request.document_id not in pipelines:
        try:
            pipelines[request.document_id] = RAGPipeline(
                store_path=vector_path
            )
        except Exception as e:
            return {"error": f"Pipeline init failed: {str(e)}"}

    pipeline = pipelines[request.document_id]

    start = time.time()
    response = pipeline.run(request.query)
    latency = time.time() - start

    return {
        "answer": response,
        "latency": round(latency, 3)
    }


@app.get("/status/{document_id}")
def status(document_id: str):
    vector_path = os.path.join(VECTOR_BASE_DIR, document_id)

    if os.path.exists(vector_path):
        return {"status": "ready"}
    return {"status": "processing"}


@app.get("/")
def home():
    return {"message": "RAG API running"}