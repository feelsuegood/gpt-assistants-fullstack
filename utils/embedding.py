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
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_content = _file.read()
            #  file_path = f"./.cache/{files_folder}_{model_name}/
            # Create model-specific directory within temp_dir
            model_dir = os.path.join(temp_dir, f"{files_folder}_{model_name}")
            os.makedirs(model_dir, exist_ok=True)
            temp_file_path = os.path.join(model_dir, _file.name)

            # Create cache directory within temp_dir
            cache_dir = LocalFileStore(
                os.path.join(temp_dir, f"{embeddings_folder}_{model_name}")
            )

            # Save the temporary file
            with open(temp_file_path, "wb") as f:
                f.write(file_content)

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

            retriever = vectorstore.as_retriever()
            return retriever
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None
