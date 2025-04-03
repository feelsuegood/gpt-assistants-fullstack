import streamlit as st

st.set_page_config(
    page_title="FullstackGPT Home",
    page_icon="🤖",
)

with st.sidebar:
    st.title("Sidebar title")
    st.text_input("aaa")

st.title("Title")

tab_one, tab_two, tab_three = st.tabs(["A", "B", "C"])

with tab_one:
    st.write("a")

with tab_two:
    st.write("b")

with tab_three:
    st.write("c")
