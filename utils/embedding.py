from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.storage import LocalFileStore
import streamlit as st


@st.cache_resource(show_spinner="Embedding file...")
def embed_file(_file, files_folder, embeddings_folder, _embeddings_model, model_name):
    """
    Embed a file and return a retriever for the embedded content.

    Args:
        file: The file to be embedded

    Returns:
        A retriever object that can be used to search the embedded content
    """
    file_content = _file.read()
    # file_path = f"./.cache/# [x] variables -> files/{file.name}"
    file_path = f"./.cache/{files_folder}_{model_name}/{_file.name}"
    # [x]print(file_path)
    with open(file_path, "wb") as f:
        f.write(file_content)
        # cache_dir = LocalFileStore(f"./.cache/ # [x] variables -> embeddings/{file.name}")
        cache_dir = LocalFileStore(
            f"./.cache/{embeddings_folder}_{model_name}/{_file.name}"
        )
        # [x] print(cache_dir)
        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n",
            chunk_size=600,
            chunk_overlap=100,
        )
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load_and_split(text_splitter=splitter)
        # [x] variable -> OpenAIEmbeddings()
        embeddings = _embeddings_model
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            embeddings,
            cache_dir,
        )
        vectorstore = FAISS.from_documents(
            docs,
            cached_embeddings,
        )
        retriever = vectorstore.as_retriever()
        # [x] print(retriever)
        return retriever
