# build_index.py

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()  # loads OPENAI_API_KEY

DATA_DIR = "./data"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "my_rag_collection"


def load_documents():
    # Load PDFs
    pdf_loader = DirectoryLoader(
        DATA_DIR,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    pdf_docs = pdf_loader.load()

    # Load text files
    txt_loader = DirectoryLoader(
        DATA_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True,
    )
    txt_docs = txt_loader.load()

    return pdf_docs + txt_docs

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return splitter.split_documents(documents)

def build_vectorstore(splits):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    vectorstore.persist()
    return vectorstore

if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")

    splits = split_documents(docs)
    print(f"Split into {len(splits)} chunks")

    build_vectorstore(splits)
    print("Vector store built and saved.")
