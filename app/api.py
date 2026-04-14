from fastapi import FastAPI
from app.pipeline import RAGPipeline
from app.config import settings

app = FastAPI()

pipeline = RAGPipeline(store_path=settings.VECTOR_DB_PATH)

@app.get("/")
def home():
    return {"message": "RAG API running"}

@app.post("/ask")
def ask(query: str):
    response = pipeline.run(query)
    return {"answer": response}