from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

def create_vector_store(file_path: str, save_path: str):
    loader = PyPDFLoader(file_path=file_path)

    pages = list(loader.lazy_load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = splitter.split_documents(pages)

    embedding_model = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(docs, embedding_model)

    vector_store.save_local(save_path)

    print("[+] Vector store created successfully!")