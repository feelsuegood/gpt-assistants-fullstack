import json
from langchain.document_loaders import UnstructuredFileLoader
from langchain.schema.output import Generation
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser
from sympy import true

# [ ] Code challenge #1: using function calling instead of prompts
# [ ] Code challenge #2: Create a switch (enable/disable) that shows the correct answer or not,
# so that a user can choose to keep trying or not
# If enabled, a user can see the correct answer right away


class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)


output_parser = JsonOutputParser()

st.set_page_config(
    page_title="QuizGPT",
    page_icon="🤓",
)
st.title("QuizGPT")


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-4.1-nano-2025-04-14",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful assistant role-playing as a teacher.

Your task is to create 10 multiple-choice questions to test the user's knowledge based ONLY on the following context.

Instructions:
- Each question must have exactly 4 answer choices.
- Mark the correct answer by adding (o) immediately after it.
- The other three answers must be plausible but incorrect.
- Randomize the position of the correct answer among the choices.
- Focus only on information contained in the provided context.

Examples:

Question: What is the color of the ocean?
Answers: Red | Yellow | Green | Blue(o)

Question: What is the capital of Georgia?
Answers: Baku | Tbilisi(o) | Manila | Beirut

Question: When was Avatar released?
Answers: 2007 | 2001 | 2009(o) | 1998

Question: Who was Julius Caesar?
Answers: A Roman Emperor(o) | Painter | Actor | Model

Now, based on the following context, create your 10 questions and answers.

Context:
{context}
""",
        )
    ]
)

questions_chain = {"context": format_docs} | questions_prompt | llm

formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a powerful formatting algorithm that converts exam questions into clean JSON format.

Instructions:
- Each question is followed by multiple answers separated by "|".
- An answer marked with "(o)" is the correct one.
- Parse and format everything carefully into JSON.
- For each answer, set "correct": true if it has (o), otherwise "correct": false.
- Remove the (o) marker from the final answer text in the JSON output.

Example Input:

Question: What is the color of the ocean?
Answers: Red|Yellow|Green|Blue(o)

Question: What is the capital of Georgia?
Answers: Baku|Tbilisi(o)|Manila|Beirut

Question: When was Avatar released?
Answers: 2007|2001|2009(o)|1998

Question: Who was Julius Caesar?
Answers: A Roman Emperor(o)|Painter|Actor|Model

Example Output:

```json
{{
  "questions": [
    {{
      "question": "What is the color of the ocean?",
      "answers": [
        {{ "answer": "Red", "correct": false }},
        {{ "answer": "Yellow", "correct": false }},
        {{ "answer": "Green", "correct": false }},
        {{ "answer": "Blue", "correct": true }}
      ]
    }},
    {{
      "question": "What is the capital of Georgia?",
      "answers": [
        {{ "answer": "Baku", "correct": false }},
        {{ "answer": "Tbilisi", "correct": true }},
        {{ "answer": "Manila", "correct": false }},
        {{ "answer": "Beirut", "correct": false }}
      ]
    }},
    {{
      "question": "When was Avatar released?",
      "answers": [
        {{ "answer": "2007", "correct": false }},
        {{ "answer": "2001", "correct": false }},
        {{ "answer": "2009", "correct": true }},
        {{ "answer": "1998", "correct": false }}
      ]
    }},
    {{
      "question": "Who was Julius Caesar?",
      "answers": [
        {{ "answer": "A Roman Emperor", "correct": true }},
        {{ "answer": "Painter", "correct": false }},
        {{ "answer": "Actor", "correct": false }},
        {{ "answer": "Model", "correct": false }}
      ]
    }}
  ]
}}
```

Now, given the following questions, generate the corresponding JSON format:

Questions:
{context}

 """,
        ),
    ]
)

formatting_chain = formatting_prompt | llm


@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    """
    Split a file and return a retriever for the embedded content.

    Args:
        file: The file to be embedded

    Returns:
        A retriever object that can be used to search the embedded content
    """
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n",
            chunk_size=600,
            chunk_overlap=100,
        )
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load_and_split(text_splitter=splitter)
        return docs


@st.cache_data(show_spinner="Making quiz...")
# * add another parameter if your parameter isn't hashable
def run_quiz_chain(_docs, topic):
    chain = {"context": questions_chain} | formatting_chain | output_parser
    return chain.invoke(_docs)


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(
        lang="en",
        doc_content_chars_max=1000,
        top_k_results=3,
    )  # type: ignore
    return retriever.get_relevant_documents(term)


with st.sidebar:
    docs = None
    choice = st.selectbox(
        "Choose what you want to use",
        (
            "File",
            "Wikipedia Article",
        ),
    )

    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx, .txt, or .pdf file",
            [
                "docx",
                "txt",
                "pdf",
            ],
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            docs = wiki_search(topic)

if not docs:
    st.markdown(
        """
    Welcome to QuizGPT.
    
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.

    Get started by uploading a file or searching on Wikipedia in the sidebar.
"""
    )
else:
    # questions_response = questions_chain.invoke(docs)
    # st.write(questions_response.content)
    # formatting_response = formatting_chain.invoke(
    #     {"context": questions_response.content}
    # )
    # # ! don't forget '.content'
    # st.write(formatting_response.content)
    quiz_topic = topic if topic else (file.name if file is not None else "")
    response = run_quiz_chain(docs, quiz_topic)
    # st.write(response)
    with st.form("questions_form"):
        # think relations based on questions
        for question in response["questions"]:
            st.write(question["question"])
            value = st.radio(
                "Select an option",
                [answer["answer"] for answer in question["answers"]],
                index=None,
            )
            # st.json({"value": value, "correct": true} in question["answers"])
            if {"answer": value, "correct": true} in question["answers"]:
                st.success("Correct 🙆🏻‍♀️")
            elif value is not None:
                for answer in question["answers"]:
                    if answer["correct"] == true:
                        correct_answer = answer["answer"]
                st.error(f"Incorrect 🙅🏻‍♀️. \n\nThe answer is '{correct_answer}'")
        button = st.form_submit_button()
