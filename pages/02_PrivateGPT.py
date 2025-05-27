from langchain_ollama import ChatOllama
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationSummaryBufferMemory
import streamlit as st
import requests
from utils.embedding_local import embed_local_file


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

if "previous_model" not in st.session_state:
    st.session_state["previous_model"] = "mistral"
    st.session_state["llm"] = ChatOllama(
        model="mistrral:latest",
        temperature=0.1,
        callbacks=[ChatCallBackHandler()],
    )

if "memory" not in st.session_state:
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
if "uploaded_file" not in st.session_state:
    st.session_state["uploaded_file"] = None


memory = st.session_state["memory"]
selected_model = st.session_state["previous_model"]


# change llm model
def change_llm_model(model_name):
    return ChatOllama(
        model=model_name,
        temperature=0.1,
        callbacks=[ChatCallBackHandler()],
        verbose=True,
    )


# change embeddings model
def change_embeddings_model(model_name):
    return OllamaEmbeddings(
        model=model_name,
    )


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
You are a concise AI assistant. Follow these rules strictly:
1. Use ONLY the provided context to answer
2. If information is not in the context, say "I don't know"
3. Keep responses brief and focused (2-3 sentences maximum)
4. Never repeat previous conversations
5. Never mention that you're using any context or history

Context:
{context}

Previous conversation (for reference only, do not mention):
{history}

Question: {question}
Answer: """
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
    selected_model = st.selectbox(
        "Select a model",
        options=["mistral", "gemma3"],
        key="model_selector",
    )
    st.markdown(
        f"You selected: 🤖&nbsp;{selected_model}\n\n"
        "**⚠️ Be careful that you lose your conversation history when changing the model.**"
    )
    if st.session_state["previous_model"] != st.session_state["model_selector"]:
        # Change llm model and clear embedding cache
        embed_local_file.clear()
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
    retriever = embed_local_file(
        file,
        "private_files",
        "private_embeddings",
        OllamaEmbeddings(model="mistral"),
        selected_model,
    )
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
