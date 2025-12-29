import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_ollama import ChatOllama 
from langchain_community.embeddings import OllamaEmbeddings

load_dotenv()

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "my_rag_collection"

def get_vectorstore():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")  # or another embedding model in Ollama
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

def build_rag_chain():
    llm = ChatOllama(
        model="llama3.1:8b",  # or any model you pulled in Ollama
        temperature=0.0,
    )
    vectorstore = get_vectorstore()          # same Chroma index as before
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5}, search_type="similarity")

    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.
Answer ONLY using the information in the context.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}
""")

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain
# def get_vectorstore():
#     embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
#     return Chroma(
#         collection_name=COLLECTION_NAME,
#         embedding_function=embeddings,
#         persist_directory=CHROMA_DIR,
#     )

# def build_rag_chain():
#     llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

#     vectorstore = get_vectorstore()
#     retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

#     prompt = ChatPromptTemplate.from_template("""
# You are a helpful assistant.
# Answer ONLY using the information in the context.
# If the answer is not in the context, say you don't know.

# Context:
# {context}

# Question:
# {question}
# """)

#     # 1) Retrieve docs
#     # 2) Format them into the prompt
#     # 3) Call LLM
#     # 4) Parse text
#     rag_chain = (
#         {"context": retriever, "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )

#     return rag_chain

if __name__ == "__main__":
    rag_chain = build_rag_chain()

    while True:
        question = input("\nAsk a question (or 'exit'): ")
        if question.lower() in {"exit", "quit"}:
            break

        answer = rag_chain.invoke(question)
        print("\nAnswer:\n", answer)
