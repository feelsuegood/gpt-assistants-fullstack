# GPT Assistansts 🤖💁‍♀️

A multi-purpose AI assistants platform built with LangChain and Streamlit.  
This project enables document Q&A, private file chat, quiz generation, and more, all running on your machine with support for both OpenAI and local models.

---

## ✨ Features

- **DocumentGPT** 📄:  
  Upload `.txt`, `.pdf`, or `.docx` files and chat with an AI about their content.  
  Session memory is used to provide context-aware answers.

- **PrivateGPT** 🔒:  
  Chat privately with your own files using local LLMs (e.g., Mistral, Qwen).  
  Choose between multiple models and keep your data on your device.

- **QuizGPT** 📝:  
  Generate multiple-choice quizzes from uploaded files or Wikipedia articles.  
  Test your knowledge interactively and get instant feedback.

- **SiteGPT** 📝:  
  Ask anything about a website — SiteGPT loads the site's sitemap, extracts the content, and answers your questions accurately.

- **MeetingGPT** 📝:  
  Upload a video and instantly get a transcript, a summary, and a chatbot to answer your questions.

- **InvestorGPT** 📝:
  [Agent Types](https://python.langchain.com/v0.1/docs/modules/agents/agent_types/)
- **Extensible** 🧩:  
  Easily add new GPT-powered tools (e.g., InvestorGPT) by creating new Streamlit pages.

---

## 🗂️ Project Structure

```
.
├── pages/           # Streamlit app pages (DocumentGPT, PrivateGPT, QuizGPT, etc.)
├── utils/           # Utility modules (e.g., embedding functions)
├── practice/        # Example notebooks and experiments
├── .cache/          # Cached files, embeddings, and quiz data
├── requirements.txt # Python dependencies
├── Home.py          # Main Streamlit entry point
└── README.md        # This file
```

---

## 🚀 Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/feelsuegood/gpt-assistants.git
   cd gpt-assistants
   ```

2. **Set up a virtual environment**

   ```bash
   python3.11 -m venv env
   source env/bin/activate
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
   - (Optional) Set up a Pinecone account for vector storage.

6. **Run the app**
   ```bash
   streamlit run Home.py
   ```

---

## 🧠 Model Information

- **Ollama Local Models** 🦙

  - **Mistral** and **Qwen (qwen:0.5b)** are used via the [Ollama](https://ollama.com/) local model server.
  - Make sure to pull these models with `ollama pull mistral` and `ollama pull qwen:0.5b` before running PrivateGPT.

- **OpenAI GPT** (for cloud-based features) ☁️

- **Orca Mini 3B** (`orca-mini-3b-gguf2-q4_0.gguf`) 🐳

  - Developer: Microsoft
  - License: CC BY-NC-SA 4.0 (non-commercial research only)

---

## 📒 Notebooks & Examples

- See the `practice/` folder for Jupyter notebooks demonstrating core features and experiments.

---

## 📚 References

- [Streamlit Documentation](https://docs.streamlit.io/develop/api-reference)
- [Nomad Coders fullstack GPT](https://nomadcoders.co/fullstack-gpt)

---

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

---

## 🏷️ License

This project is for non-commercial research and educational purposes only.  
See individual model licenses for details.

## More

- Use [langsmith](https://smith.langchain.com/) to trace langchain output