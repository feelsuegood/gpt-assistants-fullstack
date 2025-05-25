import streamlit as st

st.set_page_config(
    page_title="GPT Assistants",
    page_icon="🤖",
)

# * Streamlit provies auth for the app
# * But can check if the required API keys are in the secrets.toml file
# If not, we can show an error message
# # Check required API keys
# required_api_keys = {
#     "OPENAI_API_KEY": "OpenAI API key",
#     "LANGCHAIN_API_KEY": "LangChain API key",
#     "HUGGINGFACEHUB_API_TOKEN": "Hugging Face API token",
#     "ALPHA_VANTAGE_API_KEY": "Alpha Vantage API key",
#     "PINECONE_API_KEY": "Pinecone API key",
# }

# missing_keys = []
# for key, name in required_api_keys.items():
#     # Check if the key is in the secrets.toml file
#     if key not in st.secrets:
#         missing_keys.append(name)


# if missing_keys:
#     st.error(
#         f"""⚠️ Missing required API keys:

# {', '.join(missing_keys)}

# Please add these keys to your .streamlit/secrets.toml file to use all features."""
#     )
# else:
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
    st.link_button("📌 SiteGPT", "SiteGPT", use_container_width=True)
with col2:
    st.link_button("📆 MeetingGPT", "MeetingGPT", use_container_width=True)
    st.link_button("📈 InvestorGPT", "InvestorGPT", use_container_width=True)
    st.link_button("🚧 AssistantAPI", "AssistantAPI", use_container_width=True)
    st.link_button("🚧 FileAssistantAPI", "FileAssistantAPI", use_container_width=True)
