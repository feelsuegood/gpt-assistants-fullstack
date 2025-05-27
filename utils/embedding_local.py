import os
from langchain_unstructured import UnstructuredLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.storage import LocalFileStore
import streamlit as st


@st.cache_resource(show_spinner="Embedding file...")
def embed_local_file(
    _file, files_folder, embeddings_folder, _embeddings_model, model_name
):
    """
    Embed a file and return a retriever for the embedded content.

    Args:
        file: The file to be embedded

    Returns:
        A retriever object that can be used to search the embedded content
    """
    file_content = _file.read()
    # create cache directory path
    cache_base_dir = "./.cache"
    files_dir = f"{cache_base_dir}/{files_folder}_{model_name}"
    embeddings_dir = f"{cache_base_dir}/{embeddings_folder}_{model_name}"

    # create necessary directories
    os.makedirs(cache_base_dir, exist_ok=True)
    os.makedirs(files_dir, exist_ok=True)
    os.makedirs(embeddings_dir, exist_ok=True)

    # file_path = f"./.cache/# [x] variables -> files/{file.name}"
    # file_path = f"./.cache/{files_folder}_{model_name}/{_file.name}"
    # [x]print(file_path)

    # save file path
    file_path = os.path.join(files_dir, _file.name)
    # Check if embeddings already exist for this file and model
    if os.path.exists(file_path):
        # If file content is the same, use existing embeddings
        with open(file_path, "rb") as f:
            if f.read() == file_content:
                cache_dir = LocalFileStore(embeddings_dir)
                if os.path.exists(embeddings_dir) and os.listdir(embeddings_dir):
                    # Load existing vectorstore
                    vectorstore = FAISS.load_local(
                        embeddings_dir,
                        _embeddings_model,
                        allow_dangerous_deserialization=True,
                        #! allow deserialization from local file
                        #! allow_dangerous_deserialization=True is a security risk
                        #! only use in local environment and safe to use
                    )
                    return vectorstore.as_retriever()

    # If file doesn't exist or content is different, create new embeddings
    with open(file_path, "wb") as f:
        f.write(file_content)
        # cache_dir = LocalFileStore(f"./.cache/ # [x] variables -> embeddings/{file.name}")
        # cache_dir = LocalFileStore(
        #     f"./.cache/{embeddings_folder}_{model_name}/{_file.name}"
        # )
        cache_dir = LocalFileStore(embeddings_dir)

        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n",
            chunk_size=600,
            chunk_overlap=100,
        )
        loader = UnstructuredLoader(file_path)
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

        # save vectorstore for future use
        vectorstore.save_local(embeddings_dir)

        retriever = vectorstore.as_retriever()

        return retriever
