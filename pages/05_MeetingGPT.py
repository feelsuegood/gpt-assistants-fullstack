import subprocess
from altair import Text
from pydub import AudioSegment
import math
import openai
import glob
import streamlit as st
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser
from langchain.vectorstores.faiss import FAISS
from langchain.storage import LocalFileStore
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings


llm = ChatOpenAI(
    temperature=0.1,
)
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=800,
    chunk_overlap=100,
)

# kill switch
has_transcript = os.path.exists("./.cache/astrology.txt")


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file_path):
    file_name = os.path.basename(file_path)
    cache_dir = LocalFileStore(
        f"./.cache/meeting_embeddings/{file_name}",
    )
    loader = TextLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    # [x] variable -> OpenAIEmbeddings()
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
    if has_transcript:
        return
    audio_path = video_path.replace("mp4", "mp3")
    command = ["ffmpeg", "-y", "-i", video_path, "-vn", audio_path]
    subprocess.run(command)


@st.cache_data()
def cut_audio_in_chunks(audio_path, chunk_size, chunks_folder):
    if has_transcript:
        return
    track = AudioSegment.from_mp3(audio_path)
    chunk_leng = chunk_size * 60 * 1000
    chunks = math.ceil(len(track) / chunk_leng)
    for i in range(chunks):
        start_time = i * chunk_leng
        end_time = (i + 1) * chunk_leng
        chunk = track[start_time:end_time]
        chunk.export(f"{chunks_folder}/chunk_{i}.mp3", format="mp3")


@st.cache_data()
def transcribe_chunks(chunks_folder, transcript_path):
    if has_transcript:
        return
    files = glob.glob(f"{chunks_folder}/*.mp3")
    files.sort()
    for file in files:
        with open(file, "rb") as audio_file, open(transcript_path, "a") as text_file:
            transcript = openai.Audio.transcribe(
                "whisper-1",
                audio_file,
                language="en",
            )
            text_file.write(transcript["text"])  # type: ignore


st.set_page_config(page_title="MeetingGPT", page_icon="📆")

# st.balloons()

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
    chunks_folder = "./.cache/chunks"
    with st.status("Loading the video...") as status:
        video_content = video.read()
        video_path = f"./.cache/{video.name}"
        audio_path = video_path.replace("mp4", "mp3")
        transcript_path = video_path.replace("mp4", "txt")
        with open(video_path, "wb") as f:
            f.write(video_content)
        status.update(label="Extrating the audio...")
        extract_audio_from_video(
            video_path,
        )
        status.update(label="Extrating the audio...")
        cut_audio_in_chunks(audio_path, 5, chunks_folder)
        status.update(label="Transcribing the audio...")
        transcribe_chunks(chunks_folder, transcript_path)

    transcript_tab, summary_tab, qa_tab = st.tabs(
        [
            "Transcript",
            "Summary",
            "Q&A",
        ]
    )

    with transcript_tab:
        with open(transcript_path, "r") as file:
            st.write(file.read())

    with summary_tab:
        start = st.button("Generate Summary")
        if start:
            loader = TextLoader(transcript_path)
            splitter = splitter
            docs = loader.load_and_split(text_splitter=splitter)

            first_summary_prompt = ChatPromptTemplate.from_template(
                """
                Write a concise summary of the following:
                "{text}"
                CONCISE SUMMARY:
                """
            )

            # first_summary_chain = first_summary_prompt|llm
            # summary = first_summary_chain.invoke({"text": docs[0].page_content}).content
            # * don't have to add ".conentent"
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
                    # intermediate summary
                    st.write(summary)
            st.write(summary)

    with qa_tab:

        retriever = embed_file(transcript_path)

        docs = retriever.invoke("How the birth chart affects someone's personality?")

        st.write(docs)
