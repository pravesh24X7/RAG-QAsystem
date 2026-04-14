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

def is_valid_vector_store(path):
    return (
        os.path.exists(path) and
        os.path.isfile(os.path.join(path, "index.faiss")) and
        os.path.isfile(os.path.join(path, "index.pkl"))
    )


def get_retriever(store_path: str="", file_path :str=""):

    if not store_path:
        raise ValueError("store_path is required")

    if not is_valid_vector_store(store_path):
        print("[INFO] Vector store not found. Creating...")

        from .indexing import create_vector_store

        create_vector_store(
            file_path=file_path,
            save_path=store_path
        )

    embedding_model = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.load_local(
        store_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "lambda_mult": 0.5,
        }
    )

    return retriever