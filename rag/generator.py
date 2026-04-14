from langchain_groq import ChatGroq
from langchain_core.prompts import load_prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from app.config import settings

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def execution_chain(retriever):
    llm_model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",
                     temperature=0.5,
                     model_kwargs={},
                     streaming=True)
    
    prompt = load_prompt(f"{settings.PROMPT_PATH}/prompt.json")
    parser = StrOutputParser()

    chain = ({
        "context": retriever | RunnableLambda(format_docs),
        "query": RunnablePassthrough()
    }) | prompt | llm_model | parser

    return chain