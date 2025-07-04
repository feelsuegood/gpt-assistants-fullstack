import httpx
import re
from fake_useragent import UserAgent
from langchain_community.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st

# * https://openai.com/sitemap.xml
# * https://www.google.com/forms/sitemaps.xml

st.set_page_config(
    page_title="SiteGPT",
    page_icon="📌",
)

# Initialize a UserAgent object
ua = UserAgent()
client = httpx.Client(
    headers={
        "User-Agent": ua.random,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }
)

llm = ChatOpenAI(temperature=0.1)

answers_prompt = ChatPromptTemplate.from_template(
    """
You are tasked with answering the user's question using ONLY the provided context. Follow these rules strictly:

1. If the context contains the answer, provide it clearly.
2. If the context does not contain the answer, reply: "I don't know."
3. After your answer, always assign a score from 0 to 5:
   - 5 = Fully answers the question
   - 0 = No relevant information
   - Partial answers = Score appropriately between 1 and 4
4. Always include both the Answer and the Score.

Context:
{context}

Examples:

Question: How far away is the moon?
Answer: The moon is 384,400 km away.
Score: 5

Question: How far away is the sun?
Answer: I don't know
Score: 0

Now, your turn:

Question: {question}
"""
)


# 1st chain: rank answer for every page
def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm

    # return a dictionary that has a question and  a list of answer dictioneries and question
    answers = {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {
                        "context": doc.page_content,
                        "question": question,
                    }
                ).content,
                "source": doc.metadata.get("source", "Unknown source"),
                "date": doc.metadata.get(
                    "lastmod", "No date available"
                ),  # default value
            }
            for doc in docs
        ],
    }
    return answers


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to respond to the user's question.

            - Prioritize answers with the highest score (i.e., more helpful answers).
            - Among equally scored answers, favor the most recent ones.
            - You MUST cite the original source exactly as provided for each answer you use.
            - Format the final answer like this:
                Answer: [your answer]  
                Sources: [source of the answer]  
                Date: [date of the answer ]
            --------------------------------------------
            Pre-existing Answers:
            {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date'].split('T')[0]}\n"
        for answer in answers
    )
    chosen_answer = choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )
    return chosen_answer


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    text = str(soup.get_text())

    # 1. Remove all "opens in a new window"
    text = re.sub(r"\(opens in a new window\)", "", text)
    # 2.Delete duplicate menu keywords, use pattern capture
    for menu in [
        "Switch to",
        "ChatGPT",
        "Sora",
        "API Platform",
        "Research",
        "For Business",
        "Stories",
        "Company",
        "News",
    ]:
        text = text.replace(menu, "")

    # 3. Remove space
    clean_text = re.sub(r"\s+", " ", text).strip()
    return clean_text


# if same url, the function doesn't run again
@st.cache_resource(show_spinner="Loading a website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    # Session setup to disable SSL verification
    verify_ssl = False
    loader = SitemapLoader(
        url,
        header_template={
            "User-Agent": ua.random,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        },
        parsing_function=parse_page,
        requests_per_second=3,
        verify_ssl=verify_ssl,
    )
    docs = loader.load_and_split(text_splitter=splitter)
    # option: cache embeddings - should create a folder for each URL firstly
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vector_store.as_retriever()


st.markdown(
    """          
    # SiteGPT
       
     Ask questions about the content of a website.
             
     Start by writing the URL of the website on the sidebar.

     ### Example
    - Site: https://www.google.com/forms/sitemaps.xml
    - Question: How can I create a form?
 """
)

with st.sidebar:
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com/sitemap.xml",
    )

if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please writhe down a sitemap URL")
    else:
        retriever = load_website(url)

        query = st.text_input("Ask a question to the website.")
        if query:
            chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(get_answers)
                | RunnableLambda(choose_answer)
            )

            result = chain.invoke(query)
            st.write(result.content)
