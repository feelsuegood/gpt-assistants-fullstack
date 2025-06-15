import streamlit as st

st.set_page_config(
    page_title="GPT Assistants",
    page_icon="🤖",
)

# * Streamlit provies auth for the app
# Initialize session state for API keys if not exists
if "api_keys" not in st.session_state:
    st.session_state.api_keys = {}

# Define required API keys
required_api_keys = {
    "OPENAI_API_KEY": "OpenAI API key",
    "LANGCHAIN_API_KEY": "LangChain API key",
    "HUGGINGFACEHUB_API_TOKEN": "Hugging Face API token",
    "ALPHA_VANTAGE_API_KEY": "Alpha Vantage API key",
    "PINECONE_API_KEY": "Pinecone API key",
}

# Create sidebar for API key inputs
with st.sidebar:
    st.header("🔑 API Keys Configuration")

    for key, name in required_api_keys.items():
        # Try to get from secrets first, if not available use session state
        default_value = st.secrets.get(key, st.session_state.api_keys.get(key, ""))
        api_key = st.text_input(
            f"Enter your {name}",
            value=default_value,
            type="password",
            key=f"input_{key}",
        )
        if api_key:
            st.session_state.api_keys[key] = api_key

# Check if all required API keys are present
missing_keys = [
    name
    for key, name in required_api_keys.items()
    if not st.session_state.api_keys.get(key)
]

if missing_keys:
    st.warning(
        f"""⚠️ Missing required API keys:

{', '.join(missing_keys)}

Please enter your API key in the left sidebar. The key you enter will persist throughout your browser session."""
    )
else:
    st.balloons()

    st.markdown(
        """
    # 🤖 Welcome to my GPT Assistants

    **This is a comprehensive AI-powered assistant platform that demonstrates various applications of Large Language Models (LLMs) in real-world scenarios.**

    - **📄 DocumentGPT**: Chat with your documents using OpenAI's GPT
    - **🔒 PrivateGPT (Local Only)**: Secure document analysis using local LLMs (Mistral, Gemma3)
    - **💡 QuizGPT**: AI-powered quiz generation from any content
    - **📌 SiteGPT**: Intelligent website analysis and Q&A system
    - **📆 MeetingGPT**: Automated meeting transcription and summarization
    - **📈 InvestorGPT**: AI-powered investment research assistant

    ### 🚧 Coming Soon
    - **👨‍🍳 ChefGPT**: Recipe recommendations and cooking assistant
    - **🔧 AssistantAPI**: OpenAI Assistants API implementation for investment research
    - **📁 FileAssistantAPI**: File handling with OpenAI Assistants API

    **Feel free to explore each application and see AI assistants in action!**

    """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.link_button("📄 DocumentGPT", "DocumentGPT", use_container_width=True)
        st.link_button(
            "🔒 PrivateGPT (Local Only)", "PrivateGPT", use_container_width=True
        )
        st.link_button("💡 QuizGPT", "QuizGPT", use_container_width=True)

    with col2:
        st.link_button("📌 SiteGPT", "SiteGPT", use_container_width=True)
        st.link_button("📆 MeetingGPT", "MeetingGPT", use_container_width=True)
        st.link_button("📈 InvestorGPT", "InvestorGPT", use_container_width=True)

    with col3:
        st.link_button("⚠️ 🍳 ChefGPT", "ChefGPT", use_container_width=True)
        st.link_button(
            "⚠️ 📊 AssistantsAPI",
            "AssistantsAPI",
            use_container_width=True,
        )
        st.link_button(
            "⚠️ 📚 FileAssistantsAPI",
            "FileAssistantsAPI",
            use_container_width=True,
        )
