# RAG-QAsystem
RAG-Based Question Answering System


### Project Structure
RAG-QAsystem/
│
├── app/
│   ├── main.py              # entry point (CLI / API)
│   ├── pipeline.py          # full RAG pipeline
│   ├── config.py            # configs (paths, models)
│
├── rag/
│   ├── indexing.py
│   ├── retriever.py
│   ├── generator.py         # LLM logic
│
├── prompts/
│   └── prompt.json
│
├── data/
│   └── sample.pdf
│
├── vector_store/
│
├── utils/
│   ├── logger.py
│   └── helpers.py
│
├── requirements.txt
└── README.md