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

st.set_page_config(
    page_title="MeetingGPT",
    page_icon="📆",
)

openai.api_type = "openai"

# Initialize session_state
if "transcript_path" not in st.session_state:
    st.session_state.transcript_path = None

if "previous_video_name" not in st.session_state:
    st.session_state["previous_video_name"] = None

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


st.markdown(
    """
# MeetingGPT
             
Upload a video and I will give you a transcript, a summary and a chat bot to ask any questions about it.
 
Get started by uploading a video file in the sidebar.
"""
)

with st.sidebar:
    video = st.file_uploader("Video", type=["mp4", "avi", "mkv", "mov"])

if video:
    # Compare current video name with previous video name
    if video.name != st.session_state["previous_video_name"]:
        st.session_state["previous_video_name"] = video.name
        st.session_state.transcript_path = None  # Reset transcript path for new video

    # If the transcript has already been processed, skip the processing
    if not st.session_state.transcript_path:
        with st.status("Loading the video...") as status:
            # Create temporary file path
            video_path = os.path.join(TEMP_DIR, video.name)
            transcript_path = os.path.join(TEMP_DIR, video.name.replace("mp4", "txt"))
            st.session_state.transcript_path = transcript_path

            # Save video file
            with open(video_path, "wb") as f:
                f.write(video.read())

            status.update(label="Extracting the audio...")
            audio_path = extract_audio_from_video(video_path)

            status.update(label="Cutting audio into chunks...")
            chunk_paths = cut_audio_in_chunks(audio_path, 5)

            status.update(label="Transcribing the audio...")
            transcribe_chunks(chunk_paths, transcript_path)

    # transcript_tab, summary_tab, qa_tab = st.tabs(["Transcript", "Summary", "Q&A"])

    with st.container(border=True):
        tab = st.radio(
            "Select a view", ["Transcript", "Summary", "Q&A"], horizontal=True
        )

    # with transcript_tab:
    if tab == "Transcript":
        with open(st.session_state.transcript_path, "r") as file:
            st.write(file.read())

    # with summary_tab:
    if tab == "Summary":
        start = st.button("Generate Summary")
        if start:
            loader = TextLoader(st.session_state.transcript_path)
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

    # with qa_tab:
    if tab == "Q&A":
        # Initialize
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Container for chat history
        chat_container = st.container()

        # Input at the bottom
        question = st.chat_input("Ask a question about the video...")

        # Show chat history in the container
        with chat_container:
            for message in st.session_state.messages:
                try:
                    role = message.get("role", "user")
                    content = message.get("content", "")
                    with st.chat_message(role):
                        st.markdown(content)
                except Exception as e:
                    st.error("Error displaying message")
                    continue

        # Handle user input
        if question:
            # Add user message
            user_message = {"role": "user", "content": question}
            st.session_state.messages.append(user_message)
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(question)

                # Generate and show AI response
                with st.chat_message("assistant"):
                    retriever = embed_file(st.session_state.transcript_path)
                    context = retriever.invoke(question)
                    context_text = "\n\n".join(doc.page_content for doc in context)

                    prompt = ChatPromptTemplate.from_template(
                        """
                    Answer the question using ONLY the following context. If you cannot find the answer in the context, say "I cannot answer this question based on the video content."

                    Context: {context}

                    Question: {question}
                    """
                    )

                    chain = prompt | llm | StrOutputParser()
                    response = chain.invoke(
                        {"context": context_text, "question": question}
                    )

                    st.markdown(response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )
