from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationSummaryBufferMemory
import streamlit as st
from utils.embedding import embed_file


# Get the API key saved in Home.py and use it
openai_api_key = st.session_state.api_keys.get("OPENAI_API_KEY")
if not openai_api_key:
    st.error("Please enter your OpenAI API key in the home page")
    st.stop()


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


llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
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
        max_token_limit=1000,
        memory_key="history",
        return_messages=True,
        input_key="question",
        output_key="text",
    )

memory = st.session_state["memory"]


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


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful AI assistant that answers questions about documents. Follow these rules:
1. Use ONLY the provided context and conversation history to answer
2. If you don't know the answer, say "I don't know"
3. Do not repeat previous answers unless specifically asked
4. Keep your answers focused and concise
5. If the question has been asked before, refer to your previous answer and add any new relevant information

Context: {context}
Previous Conversation: {history}
""",
        ),
        ("human", "{question}"),
    ]
)

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📄",
)


st.markdown(
    """
Welcome!

Use this chatbot to ask a question to an AI about your files!

Upload your files on the sidebar.
"""
)

with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt, .pdf, or .docx file", type=["txt", "pdf", "docx"]
    )

if file:
    # * Code Challenge: apply memory by using session state
    # Initializing memory when the file changes and don't drag the history about the old file to the new file.
    # New document → new context → new memory
    if file.name != st.session_state["previous_file_name"]:
        st.session_state["previous_file_name"] = file.name
        st.session_state["memory"] = ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=1000,
            memory_key="history",
            return_messages=True,
            input_key="question",
            output_key="text",
        )
    retriever = embed_file(file, "files", "embeddings", OpenAIEmbeddings(), "openai")
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
