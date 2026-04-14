import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from .indexing import create_vector_store
from app.config import settings


load_dotenv()

def load_vector_store(directory=""):
    if not directory:
        raise ValueError("Vector store directory missing")

    if not os.path.exists(directory):
        raise FileNotFoundError(
            f"Vector store not found at: {directory}\n"
        )

    embedding_model = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.load_local(
        directory,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )

    return vector_store

def get_retriever(store_path=None):
    create_vector_store(file_path=f"{settings.DATA_PATH}/Paperid_637_manuscript.pdf",
                            save_path=store_path)
    vector_store = load_vector_store(store_path)
    retriever = vector_store.as_retriever(search_type="mmr",
                                          search_kwargs={
                                              "k": 5,
                                              "lambda_mult": 0.5,
                                          })
    
    return retriever