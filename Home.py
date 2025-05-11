import streamlit as st


st.set_page_config(
    page_title="GPT Assistants",
    page_icon="🤖",
)

st.balloons()

st.markdown(
    """
# Hello!

Welcome to my GPT Assistants Fullstack Portfolio Site.

Here are the apps I've made:

"""
)


col1, col2 = st.columns(2)

with col1:
    st.link_button("📄 DocumentGPT", "DocumentGPT", use_container_width=True)
    st.link_button("🔒 PrivateGPT", "PrivateGPT", use_container_width=True)
    st.link_button("💡 QuizGPT", "QuizGPT", use_container_width=True)
with col2:
    st.link_button("📌 SiteGPT", "SiteGPT", use_container_width=True)
    st.link_button("📆 MeetingGPT", "MeetingGPT", use_container_width=True)
    st.link_button("📈 InvestorGPT", "InvestorGPT", use_container_width=True)
