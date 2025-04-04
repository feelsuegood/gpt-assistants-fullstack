import streamlit as st

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📄",
)

st.title("DocumentGPT")

with st.chat_message("human"):
    st.write("Hey mate")

with st.chat_message("ai"):
    st.write("Yes mate")

st.chat_input("Send a message to AI")
