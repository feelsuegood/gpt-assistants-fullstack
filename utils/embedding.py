from langchain_unstructured import UnstructuredLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.storage import LocalFileStore
import streamlit as st
import tempfile
import os


@st.cache_resource(show_spinner="Embedding file...")
def embed_file(_file, files_folder, embeddings_folder, _embeddings_model, model_name):
    """
    Embed a file and return a retriever for the embedded content.
    Uses temporary directory for file processing in Streamlit Cloud.
    """
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        file_content = _file.read()
        # Create model-specific directory within temp_dir
        temp_files_dir = os.path.join(temp_dir, f"{files_folder}_{model_name}")
        temp_embeddings_dir = os.path.join(
            temp_dir, f"{embeddings_folder}_{model_name}"
        )

        # Create necessary directories
        os.makedirs(temp_files_dir, exist_ok=True)
        os.makedirs(temp_embeddings_dir, exist_ok=True)

        # Define file path
        temp_file_path = os.path.join(temp_files_dir, _file.name)

        # Check if embeddings already exist for this file and model
        if os.path.exists(temp_file_path):
            # If file content is the same, use existing embeddings
            with open(temp_file_path, "rb") as f:
                if f.read() == file_content:
                    cache_dir = LocalFileStore(temp_embeddings_dir)
                    if os.path.exists(temp_embeddings_dir) and os.listdir(
                        temp_embeddings_dir
                    ):
                        # Load existing vectorstore
                        vectorstore = FAISS.load_local(
                            temp_embeddings_dir, _embeddings_model
                        )
                        return vectorstore.as_retriever()

        # If file doesn't exist or content is different, create new embeddings
        with open(temp_file_path, "wb") as f:
            f.write(file_content)

        cache_dir = LocalFileStore(temp_embeddings_dir)

        # Process the documents
        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n",
            chunk_size=600,
            chunk_overlap=100,
        )

        loader = UnstructuredLoader(
            temp_file_path,
            strategy="fast",  # fast strategy
            mode="elements",  # default mode
        )
        docs = loader.load_and_split(text_splitter=splitter)

        # Embedding processing (performed in memory)
        embeddings = _embeddings_model
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            embeddings,
            cache_dir,
        )
        vectorstore = FAISS.from_documents(
            docs,
            cached_embeddings,
        )

        # Save vectorstore for future use
        vectorstore.save_local(temp_embeddings_dir)

        retriever = vectorstore.as_retriever()
        return retriever
