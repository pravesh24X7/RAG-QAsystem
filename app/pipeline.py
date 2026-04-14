from rag.retriever import get_retriever
from rag.generator import execution_chain
from utils.prompt_generator import generate_prompt
from .config import settings


class RAGPipeline:
    def __init__(self, store_path: str, file_path: str):
        self.retriever = get_retriever(store_path=store_path, file_path=f"{settings.DATA_PATH}/{file_path}")

        print("[+] Generating prompt template...\n")
        generate_prompt(location=f"{settings.PROMPT_PATH}/prompt.json")
        
        self.chain = execution_chain(self.retriever)
    
    def run(self, query: str) -> str:
        try:

            response = self.chain.invoke(query)
            return response
        except Exception as e:
            return f"[Error Occured]:\t{str(e)}"