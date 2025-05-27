# 🤖 GPT Assistants Fullstack

Built a comprehensive AI assistant platform using OpenAI GPT, LangChain, and Streamlit that enables document analysis, private chat, quiz generation, meeting summarization, and investment analysis through eight specialized GPT applications.

## 🎯 Overview

This project showcases different AI assistants for various use cases:

- 📄 **DocumentGPT**: Analyze and interact with documents
- 🔒 **PrivateGPT**: Secure, private conversations with AI
- 💡 **QuizGPT**: AI-powered quiz generation
- 📌 **SiteGPT**: Website analysis and interaction
- 📆 **MeetingGPT**: AI assistant for meeting management
- 📈 **InvestorGPT**: AI-powered investment analysis
- 🔧 **AssistantAPI**: OpenAI Assistant API implementation
- 📁 **FileAssistantAPI**: File handling with OpenAI Assistant API


## 🛠️ Technologies

- **Frontend**: Streamlit
- **Backend**: Python, FastAPI
- **AI**: OpenAI GPT, LangChain
- **Database**: SQLite
- **File Processing**: Various document parsers (PDF, DOCX, etc.)
- **Search**: Pinecone
- **Authentication**: Built-in Streamlit auth
- **Caching**: Streamlit caching with FAISS for embeddings
- **Local Models**: Ollama for running local LLMs

## 🌟 Features

### Caching System
- Efficient caching using Streamlit's `@st.cache_resource` and `@st.cache_data`
- FAISS vector store for fast similarity search
- Model-specific embedding caches for better performance
- Automatic cache management for different LLM models

### Cloud Deployment
- Streamlit Cloud compatible
- Proper handling of temporary files and directories
- Environment-aware configurations
- Secure API key management

### Local Development
- Support for local LLM models via Ollama
- Local file caching for faster development
- Easy switching between cloud and local modes


## 🚀 Getting Started

1. **Clone the repository**
    ```bash
    git clone https://github.com/feelsuegood/gpt-assistants-fullstack.git
    cd gpt-assistants-fullstack
    ```

2. **Set up virtual environment**
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows: env\Scripts\activate
    ```

3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4. **Install Ollama (for local models)** 🦙

- Download and install Ollama from [https://ollama.com/download](https://ollama.com/download)
- Start the Ollama server:
  ```bash
  ollama serve
  ```
- Pull the required models (Mistral, Qwen):
  ```bash
  ollama pull mistral
  ollama pull qwen:0.5b
  ```

5. **Configure API keys** 🔑

- Create a `.env` file and add your OpenAI API key and any other required secrets.
- Create a `.streamlit/secrets.toml` file for Streamlit-specific configurations.
- (Optional) Set up a Pinecone account for vector storage.

6. **Run the app**
    ```bash
    streamlit run Home.py
    ```


## 🔑 Environment Variables

Required environment variables:
- `OPENAI_API_KEY`: Your OpenAI API key
- `LANGCHAIN_API_KEY`: Your Langchain API key
- `HUGGINGFACEHUB_API_TOKEN`: Your Hugging Face API token
- `ALPHA_VANTAGE_API_KEY`: Your Alpha Vantage API key
- `PINECONE_API_KEY`: Your Pinecone API key (optional)
- `LANGCHAIN_TRACING_V2`: Set to "true" for LangSmith tracing

**OpenAI GPT** (for cloud-based features) ☁️

**Ollama Local Models** 🦙
-  **Mistral** and **Gemma3** are used via the [Ollama](https://ollama.com/) local model server.
- Make sure to pull these models with `ollama pull mistral` and `ollama pull gemma3` before running PrivateGPT.

**Orca Mini 3B** (`orca-mini-3b-gguf2-q4_0.gguf`) 🐳
- Developer: Microsoft
- License: CC BY-NC-SA 4.0 (non-commercial research only)
  
### 📄 DocumentGPT
- Upload and analyze documents
- Ask questions about document content
- Extract key information
- Efficient caching of embeddings for better performance

### 🔒 PrivateGPT
- Secure, private conversations
- Local model processing with Ollama
- Data privacy focus
- Support for multiple local models (Mistral, Qwen)
- Automatic model switching with cached embeddings

### 💡 QuizGPT
- Generate quizzes from content
- Multiple question types
- Automatic grading
- Wikipedia integration for additional context

### 📌 SiteGPT
- Website analysis
- Content extraction
- Interactive Q&A

### 📆 MeetingGPT
- Meeting summarization
- Action item extraction
- Meeting analytics
- Audio transcription with OpenAI Whisper
- Efficient chunk processing for long meetings

### 📈 InvestorGPT
- Financial analysis
- Stock market data
- Investment recommendations
- Real-time market data integration


## 📝 To Do

- [x] DocumentGPT: Refactor to use the utils.embedding function for embeddings.
- [x] PrivateGPT: Refactor to use the utils.embedding function for embeddings.
- [ ] QuizGPT: Implement function calling instead of using prompt-based generation.
- [ ] QuizGPT: Add a toggle switch to show or hide the correct answers.
- [x] SiteGPT: playwright install in the virtual environment
- [ ] SiteGPT: Build a chatbot with message streaming, user and assistant roles, chat history, and memory integration.
- [ ] SiteGPT: Cache user questions and similar questions.
  - Idea: Save past questions and responses.
  - When a new question is asked, check if a similar question exists to another llm.
    - If it does, return the saved response.
    - If not, proceed with a map-reduce or rerank chain.
    - (Hint: You can implement this using function calling.)
- [ ] MeetingGPT: Finalise the Q&A tab by building a chatbot with document retrieval and answering capabilities. Use a chain to accurately respond to users' questions based on the meeting transcript.
  - Stuff Chain | ✅ Map-Reduce Chain | Map-Rerank Chain | Refine Chain
- [x] Upgrade langchain to langchain_community
- [ ] Improve stock symbol search reliability by replacing or supplementing DuckDuckGo, which can sometimes be unreliable.
- [ ] Current window link in Home.py is not possible (streamlit doesn't support)
- [ ] Add input function for OpenAI API Key and Alpha Vantage API Key from users for deployed app
- [ ] ChefGPT (Custom GPT): create user authentication function for a user to mark their favourite recipes and see their favourite recipes list 

## 🏷️ License
- CC BY-NC-SA 4.0 (non-commercial research only)
- This project is for non-commercial research and educational purposes only.  
- See individual model licenses for details.

## 📚 References
- [Streamlit Documentation](https://docs.streamlit.io/develop/api-reference)
- [Fullstack GPT - Nomad Coders](https://nomadcoders.co/fullstack-gpt)
- [Langchain Tools/Toolkits](https://python.langchain.com/docs/integrations/tools/?_gl=1*ldavbi*_ga*ODYyMjkyMzAuMTc0Njk4NjYxNw..*_ga_47WX3HKKY2*czE3NDY5ODY2ODMkbzEkZzEkdDE3NDY5ODY2OTMkajAkbDAkaDA)
- Use [langsmith](https://smith.langchain.com/) to trace langchain output
- https://docs.pydantic.dev/latest/ to use OPENAI function calling
  



