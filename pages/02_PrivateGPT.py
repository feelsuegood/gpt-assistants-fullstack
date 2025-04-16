# from langchain.chat_models import ChatOllama
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationSummaryBufferMemory
import streamlit as st

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


# init llm
def init_llm():
    model_name = st.session_state["model_selector"]
    # add ":latest" tag for mistral
    if model_name == "mistral":
        model_name = f"{model_name}:latest"
    return ChatOllama(
        model=model_name,
        temperature=0.1,
        verbose=True,
        callbacks=[ChatCallBackHandler()],
    )


# sesseion state init
if "model_selector" not in st.session_state:
    st.session_state["model_selector"] = "mistral:latest"  # default model

if "llm" not in st.session_state:
    st.session_state["llm"] = init_llm()

if "previous_file_name" not in st.session_state:
    st.session_state["previous_file_name"] = None

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationSummaryBufferMemory(
        llm=st.session_state["llm"],
        max_token_limit=150,
        return_messages=True,
    )

if "previous_model" not in st.session_state:
    st.session_state["previous_model"] = st.session_state["model_selector"]


memory = st.session_state["memory"]


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file, model_name):
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


prompt = ChatPromptTemplate.from_template(
    """
Answer the question using ONLY nothing but the following context and history.Don't use your tratining data. If you don't know the answer, just say you don't know. DON'T make anything up. 

--------------------------------------------
History: {history}
--------------------------------------------
Context: {context}
--------------------------------------------
Question:{question}"""
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
        f"You selected: 🤖&nbsp;{st.session_state['model_selector']}\n\n"
        "**⚠️ Be careful that you lose your conversation changing the model.**"
    )
    # update session state about model
    if st.session_state["previous_model"] != st.session_state["model_selector"]:
        st.session_state["previous_model"] = st.session_state["model_selector"]
        st.session_state["llm"] = init_llm()
        # initialize embedding cache
        embed_file.clear()
        # initialize file name to reproduce embedding for changed model
        st.session_state["previous_file_name"] = None

if file:
    # Initializing memory when the file changes and don't drag the history about the old file to the new file.
    # New document → new context → new memory
    if file.name != st.session_state["previous_file_name"]:
        st.session_state["previous_file_name"] = file.name
        st.session_state["memory"] = ConversationSummaryBufferMemory(
            llm=st.session_state["llm"],
            max_token_limit=150,
            return_messages=True,
        )
        st.session_state["messages"] = []  # init history

    retriever = embed_file(file, st.session_state["model_selector"])
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
                    lambda _: memory.load_memory_variables({})["history"]
                ),
            }
            | prompt
            | st.session_state["llm"]
        )

        with st.chat_message("ai"):
            # automatically st.markdown when invoke and saved by callbackhandlers
            chain.invoke(message)

        # * Optional: display memory contents
        # st.markdown("**Memory Contents:**")
        # st.json(memory.load_memory_variables({}))
else:
    st.session_state["messages"] = []
