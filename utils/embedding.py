from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
import streamlit as st


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    """
    Embed a file and return a retriever for the embedded content.

    Args:
        file: The file to be embedded

    Returns:
        A retriever object that can be used to search the embedded content
    """
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
        cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n",
            chunk_size=600,
            chunk_overlap=100,
        )
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load_and_split(text_splitter=splitter)
        embeddings = OpenAIEmbeddings()
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            embeddings,
            cache_dir,
        )
        vectorstore = FAISS.from_documents(
            docs,
            cached_embeddings,
        )
        retriever = vectorstore.as_retriever()
        return retriever
