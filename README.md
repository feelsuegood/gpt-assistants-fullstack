# Fullstack-GPT 🦙🤖

A multi-purpose, local-first AI assistant platform built with Streamlit and LangChain.  
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

- **Extensible** 🧩:  
  Easily add new GPT-powered tools (e.g., SiteGPT, MeetingGPT, InvestorGPT) by creating new Streamlit pages.

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
   git clone https://github.com/feelsuegood/fullsuetack-gpt.git
   cd fullsuetack-gpt
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

- [x] Replace embedding logic in PrivateGPT with `utils.embedding` function
- [ ] Add more GPT-powered tools (SiteGPT, MeetingGPT, InvestorGPT, etc.)
- [ ] Use function calling instead of prompts for QuizGPT
- [ ] Create a switch (enable/disable) that shows the correct answer or not, for QuizGPT

---

## 🏷️ License

This project is for non-commercial research and educational purposes only.  
See individual model licenses for details.

SiteGPT: playwright install
