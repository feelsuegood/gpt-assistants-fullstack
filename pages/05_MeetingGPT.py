import subprocess
from pydub import AudioSegment
import math
import openai
import streamlit as st
import os
import tempfile
import openai
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings

openai.api_type = "openai"

# Get the API key saved in Home.py and use it
openai_api_key = st.session_state.api_keys.get("OPENAI_API_KEY")
if not openai_api_key:
    st.error("Please enter your OpenAI API key in the home page")
    st.stop()

# Create temporary directory
TEMP_DIR = tempfile.mkdtemp()
CACHE_DIR = os.path.join(TEMP_DIR, "cache")
CHUNKS_DIR = os.path.join(TEMP_DIR, "chunks")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CHUNKS_DIR, exist_ok=True)

llm = ChatOpenAI(
    temperature=0.1,
)
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=800,
    chunk_overlap=100,
)


@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file_path):
    file_name = os.path.basename(file_path)
    cache_dir = LocalFileStore(
        os.path.join(TEMP_DIR, f"meeting_embeddings_{file_name}")
    )
    loader = TextLoader(file_path)
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


@st.cache_data()
def extract_audio_from_video(video_path):
    audio_path = os.path.join(
        TEMP_DIR, os.path.basename(video_path).replace("mp4", "mp3")
    )
    command = ["ffmpeg", "-y", "-i", video_path, "-vn", audio_path]
    subprocess.run(command)
    return audio_path


@st.cache_data()
def cut_audio_in_chunks(audio_path, chunk_size):
    track = AudioSegment.from_mp3(audio_path)
    chunk_leng = chunk_size * 60 * 1000
    chunks = math.ceil(len(track) / chunk_leng)
    chunk_paths = []
    for i in range(chunks):
        start_time = i * chunk_leng
        end_time = (i + 1) * chunk_leng
        chunk = track[start_time:end_time]
        chunk_path = os.path.join(CHUNKS_DIR, f"chunk_{i}.mp3")
        chunk.export(chunk_path, format="mp3")
        chunk_paths.append(chunk_path)
    return chunk_paths


@st.cache_data()
def transcribe_chunks(chunk_paths, transcript_path):
    with open(transcript_path, "w") as text_file:
        for file in chunk_paths:
            with open(file, "rb") as audio_file:
                transcript = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en",
                )
                text_file.write(transcript.text)


st.set_page_config(page_title="MeetingGPT", page_icon="📆")

st.markdown(
    """
# MeetingGPT
             
Welcome to MeetingGPT, upload a video and I will give you a transcript, a summary and a chat bot to ask any questions about it.
 
Get started by uploading a video file in the sidebar.
"""
)

with st.sidebar:
    video = st.file_uploader("Video", type=["mp4", "avi", "mkv", "mov"])

if video:
    with st.status("Loading the video...") as status:
        # Create temporary file path
        video_path = os.path.join(TEMP_DIR, video.name)
        transcript_path = os.path.join(TEMP_DIR, video.name.replace("mp4", "txt"))

        # Save video file
        with open(video_path, "wb") as f:
            f.write(video.read())

        status.update(label="Extracting the audio...")
        audio_path = extract_audio_from_video(video_path)

        status.update(label="Cutting audio into chunks...")
        chunk_paths = cut_audio_in_chunks(audio_path, 5)

        status.update(label="Transcribing the audio...")
        transcribe_chunks(chunk_paths, transcript_path)

    transcript_tab, summary_tab, qa_tab = st.tabs(["Transcript", "Summary", "Q&A"])

    with transcript_tab:
        with open(transcript_path, "r") as file:
            st.write(file.read())

    with summary_tab:
        start = st.button("Generate Summary")
        if start:
            loader = TextLoader(transcript_path)
            docs = loader.load_and_split(text_splitter=splitter)

            first_summary_prompt = ChatPromptTemplate.from_template(
                """
                Write a concise summary of the following:
                "{text}"
                CONCISE SUMMARY:
                """
            )

            first_summary_chain = first_summary_prompt | llm | StrOutputParser()
            summary = first_summary_chain.invoke({"text": docs[0].page_content})

            refine_prompt = ChatPromptTemplate.from_template(
                """
                You are tasked with producing a final, polished summary.

                - An existing summary has already been generated up to this point:  
                {existing_summary}

                - Additional new context is provided below:  
                ------------
                {context}
                ------------

                Your job:
                - If the new context contains useful information, **refine and improve** the existing summary by integrating the new details.
                - If the new context is irrelevant or unhelpful, **keep the original summary unchanged**.

                Important:
                - Do not remove important information from the original summary.
                - Only modify the summary when it leads to a clearer, more complete result.
                - Prefer making small, focused edits rather than rewriting the summary completely.

                Please return the final version of the summary.
                """
            )
            refine_chain = refine_prompt | llm | StrOutputParser()

            with st.status("Summarising...") as status:
                for i, doc in enumerate(docs[1:]):
                    status.update(label=f"Summarising document {i+1}/{len(docs)-1}")
                    summary = refine_chain.invoke(
                        {
                            "existing_summary": summary,
                            "context": doc.page_content,
                        }
                    )
                    st.write(summary)
            st.write(summary)

    with qa_tab:
        retriever = embed_file(transcript_path)
        docs = retriever.invoke("How the birth chart affects someone's personality?")
        st.write(docs)
