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


llm = ChatOllama(
    model="mistral:latest",
    temperature=0.1,
    # streaming=True,
    callbacks=[ChatCallBackHandler()],
)

# sesseion init
if "previous_file_name" not in st.session_state:
    st.session_state["previous_file_name"] = None

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=150,
        return_messages=True,
    )

memory = st.session_state["memory"]


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/private_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
        cache_dir = LocalFileStore(f"./.cache/private_embeddings/{file.name}")
        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n",
            chunk_size=600,
            chunk_overlap=100,
            encoding_name="cl100k_base",
        )
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load_and_split(text_splitter=splitter)
        embeddings = OllamaEmbeddings(model="mistral:latest")
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
    # ## create select box to choose a model that has a drop-down option
    # a user can chooser between mistral and qwen:0.5b (no storage issue 🥲)
    # https://docs.streamlit.io/develop/api-reference
    st.selectbox(
        "Select a model",
        options=["mistral", "qwen:0.5b"],
    )
    #
    st.write("You choose this model and this model like this")

if file:
    # Initializing memory when the file changes and don't drag the history about the old file to the new file.
    # New document → new context → new memory
    if file.name != st.session_state["previous_file_name"]:
        st.session_state["previous_file_name"] = file.name
        st.session_state["memory"] = ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=150,
            return_messages=True,
        )
    retriever = embed_file(file)
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
            | llm
        )

        with st.chat_message("ai"):
            # automatically st.markdown when invoke and saved by callbackhandlers
            chain.invoke(message)

        # * Optional: display memory contents
        # st.markdown("**Memory Contents:**")
        # st.json(memory.load_memory_variables({}))
else:
    st.session_state["messages"] = []
