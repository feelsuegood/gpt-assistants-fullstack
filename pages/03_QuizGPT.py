from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.retrievers import WikipediaRetriever

st.set_page_config(
    page_title="QuizGPT",
    page_icon="🤓",
)
st.title("QuizGPT")

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-4.1-nano-2025-04-14",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)


@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    """
    Split a file and return a retriever for the embedded content.

    Args:
        file: The file to be embedded

    Returns:
        A retriever object that can be used to search the embedded content
    """
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n",
            chunk_size=600,
            chunk_overlap=100,
        )
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load_and_split(text_splitter=splitter)
        return docs


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


with st.sidebar:
    docs = None
    choice = st.selectbox(
        "Choose what you want to use",
        (
            "File",
            "Wikipedia Article",
        ),
    )

    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx, .txt, or .pdf file",
            [
                "docx",
                "txt",
                "pdf",
            ],
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            retriever = WikipediaRetriever(
                lang="en",
                doc_content_chars_max=1000,
                top_k_results=3,
            )  # type: ignore
            with st.status("Searching Wikipedia..."):
                docs = retriever.get_relevant_documents(topic)

if not docs:
    st.markdown(
        """
    Welcome to QuizGPT.
    
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.

    Get started by uploading a file or searching on Wikipedia in the sidebar.
"""
    )
else:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a helpful assistant role-playing as a teacher.

Your task is to create 10 multiple-choice questions to test the user's knowledge based ONLY on the following context.

Instructions:
- Each question must have exactly 4 answer choices.
- Mark the correct answer by adding (o) immediately after it.
- The other three answers must be plausible but incorrect.
- Randomize the position of the correct answer among the choices.
- Focus only on information contained in the provided context.

Examples:

Question: What is the color of the ocean?
Answers: Red | Yellow | Green | Blue(o)

Question: What is the capital of Georgia?
Answers: Baku | Tbilisi(o) | Manila | Beirut

Question: When was Avatar released?
Answers: 2007 | 2001 | 2009(o) | 1998

Question: Who was Julius Caesar?
Answers: A Roman Emperor(o) | Painter | Actor | Model

Now, based on the following context, create your 10 questions and answers.

Context:
{context}
""",
            )
        ]
    )

    chain = {"context": format_docs} | prompt | llm

    start = st.button("Generate Quiz")

    if start:
        chain.invoke(docs)
