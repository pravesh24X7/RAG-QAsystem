import os

class Settings:
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vector_store/")
    MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"
    DATA_PATH = os.getenv("DATA_PATH", "./data/")
    PROMPT_PATH = os.getenv("PROMPT_PATH", "./prompt.json")


settings = Settings()