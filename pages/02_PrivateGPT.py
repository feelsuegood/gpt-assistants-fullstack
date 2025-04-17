from langchain.chat_models import ChatOllama
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationSummaryBufferMemory
import streamlit as st
import requests


st.set_page_config(
    page_title="PrivateGPT",
    page_icon="🔒",
)


class ChatCallBackHandler(BaseCallbackHandler):

    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")
        if len(st.session_state["messages"]) >= 2:
            last_human_message = st.session_state["messages"][-2]["message"]
            st.session_state["memory"].save_context(
                {"question": last_human_message},
                {"text": self.message},
            )
        self.message = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


# sesseion state init
if "previous_file_name" not in st.session_state:
    st.session_state["previous_file_name"] = None

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "llm" not in st.session_state:
    st.session_state["llm"] = ChatOllama(
        model="mistral:latest",
        temperature=0.1,
        callbacks=[ChatCallBackHandler()],
    )

if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationSummaryBufferMemory(
        llm=st.session_state["llm"],
        # If max token is too small (e.g., 250), ai repeats its answers.
        max_token_limit=1000,
        return_messages=True,
        memory_key="history",
        input_key="question",
        output_key="text",
        human_prefix="Human",
        ai_prefix="AI",
    )

if "previous_model" not in st.session_state:
    st.session_state["previous_model"] = "mistral"  # default model


memory = st.session_state["memory"]


# * replace with utils.embedding function
@st.cache_data(show_spinner="Embedding file...")
def private_embed_file(file, model_name):
    # The cached embedding and the selected model's embedding dimensions are different.
    cache_key = f"{model_name.replace(':', '_')}_{file.name}"
    file_content = file.read()
    file_path = f"./.cache/private_files/{cache_key}"
    with open(file_path, "wb") as f:
        f.write(file_content)
        cache_dir = LocalFileStore(f"./.cache/private_embeddings/{cache_key}")
        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n",
            chunk_size=600,
            chunk_overlap=100,
            encoding_name="cl100k_base",
        )
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load_and_split(text_splitter=splitter)
        embeddings = OllamaEmbeddings(
            model=model_name.replace(":latest", ""),
        )
        # print("embeddings model:", model_name)
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


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


# change llm model
def change_llm_model(model_name):
    # add ":latest" tag for mistral
    if model_name == "mistral":
        model_name = f"{model_name}:latest"
    return ChatOllama(
        model=model_name,
        temperature=0.1,
        callbacks=[ChatCallBackHandler()],
    )


#
def format_history(history):
    """Format the history to prevent the ai from repeating its answers"""
    if not history:
        return ""
    formatted_history = []
    for message in history:
        if hasattr(message, "type"):
            if message.type == "human":
                formatted_history.append(f"Human: {message.content}")
            elif message.type == "ai":
                formatted_history.append(f"AI: {message.content}")
    return "\n".join(formatted_history)


def reset_memory_and_messages():
    """Reset the conversation memory and messages"""
    st.session_state["memory"] = ConversationSummaryBufferMemory(
        llm=st.session_state["llm"],
        max_token_limit=1000,
        return_messages=True,
        memory_key="history",
        input_key="question",
        output_key="text",
        human_prefix="Human",
        ai_prefix="AI",
    )
    st.session_state["messages"] = []


prompt = ChatPromptTemplate.from_template(
    """
You are an AI assistant. Use only the following context and questions from the human to answer. Do not use your training data. 
If the answer is not in the context, say "I don't know". Do not repeat anything from the prompt or previous answers.

-----------------------------
Context:
{context}

-----------------------------
Previous questions and answers:
{history}

-----------------------------
Current question:
{question}"""
)

st.title("PrivateGPT")

st.markdown(
    """
Welcome!

This is a private chatbot that runs locally on your machine, 

allowing you to ask questions secretly about your files using AI.

Please upload your files using the sidebar.
"""
)


with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt, .pdf, or .docx file", type=["txt", "pdf", "docx"]
    )
    # * Code Challenge: select box to choose a model that has a drop-down option
    # a user can chooser between mistral and qwen:0.5b (no storage issue 🥲)
    selected_model = st.selectbox(
        "Select a model",
        options=["mistral", "qwen:0.5b"],
        key="model_selector",
    )
    st.markdown(
        f"You selected: 🤖&nbsp;{selected_model}\n\n"
        "**⚠️ Be careful that you lose your conversation history when changing the model.**"
    )
    if st.session_state["previous_model"] != st.session_state["model_selector"]:
        # Change llm model and clear embedding cache
        private_embed_file.clear()
        st.session_state["llm"] = change_llm_model(selected_model)
        st.session_state["previous_file_name"] = None
        st.session_state["previous_model"] = selected_model
        reset_memory_and_messages()
if file:
    # Check Ollama server connection
    try:
        requests.get("http://localhost:11434/api/tags")
    except requests.exceptions.ConnectionError:
        st.error(
            "Unable to connect to Ollama server. Start the server with the 'ollama serve' command."
        )
    # Initializing memory when the file changes and don't drag the history about the old file to the new file.
    # New document → new context → new memory
    if file.name != st.session_state["previous_file_name"]:
        st.session_state["previous_file_name"] = file.name
        reset_memory_and_messages()
    retriever = private_embed_file(file, selected_model)
    send_message("I'm ready. Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your files...")
    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
                "history": RunnableLambda(
                    lambda _: format_history(
                        memory.load_memory_variables({})["history"]
                    )
                ),
            }
            | prompt
            | st.session_state["llm"]
        )
        # with st.status("Generating an answer..."):
        with st.chat_message("ai"):
            # automatically st.markdown when invoke and saved by callbackhandlers
            with st.spinner("Generating an answer..."):
                chain.invoke(message)

        # print("🧠 MEMORY:", memory.load_memory_variables({})["history"])
        # print("🤖 MODEL:", st.session_state["model_selector"])
        # print("🔍 LLM:", st.session_state["llm"].model)

else:
    st.session_state["messages"] = []
