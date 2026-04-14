from app.pipeline import RAGPipeline
from app.config import settings

def main():
    pipeline = RAGPipeline(
        store_path=settings.VECTOR_DB_PATH
    )

    print("[*] RAG based QA system")

    while True:
        query = input("HUMAN:\t").strip()

        if query.lower() == "exit":
            break
        response = pipeline.run(query=query)
        print(f"AI:\t{response}")


if __name__ == "__main__":
    main()
