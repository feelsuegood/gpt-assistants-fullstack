import subprocess
from pydub import AudioSegment
import math
import openai
import glob
import streamlit as st
import os

has_transcript = os.path.exists("./.cache/astrology.txt")


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
            text_file.write(transcript["text"])


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
    chunks_folder = "./.cache/chunks"
    with st.status("Loading the video..."):
        video_content = video.read()
        video_path = f"./.cache/{video.name}"
        audio_path = video_path.replace("mp4", "mp3")
        transcript_path = video_path.replace("mp4", "txt")
        with open(video_path, "wb") as f:
            f.write(video_content)
    with st.status("Extrating the audio..."):
        extract_audio_from_video(
            video_path,
        )
    with st.status("Cutting the audio segments..."):
        cut_audio_in_chunks(audio_path, 5, chunks_folder)
    with st.status("Transcribing the audio..."):
        transcribe_chunks(chunks_folder, transcript_path)
