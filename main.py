from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY"),
)


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

index_name = "recipes"

index = pc.Index(index_name)

vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings,
)

app = FastAPI(
    title="ChefGPT. The best provider of Indian Recipes in the world",
    description="Give ChatGPT a couple of ingredients and it will give you recipes in retrun.",
    servers=[
        {
            "url": "https://editor-possibilities-baskets-sponsorship.trycloudflare.com",
        },
    ],
)


class Document(BaseModel):
    page_content: str


@app.get(
    "/recipes",
    summary="Returns a list of recipes.",
    description="Upon receiving an ingredient, this endpoint will return a list of recipes that contain that ingredient.",
    response_description="A document object that contains the recipe and preparation instructions",
    response_model=list[Document],
    openapi_extra={
        "x-openai-isConsequential": False,
        # If the x-openai-isConsequential field is true, ChatGPT treats the operation as "must always prompt the user for confirmation before running" and don't show an "always allow" button (both are features of GPTs designed to give builders and users more control over actions).
        # If the x-openai-isConsequential field is false, ChatGPT shows the "always allow button".
        # If the field isn't present, ChatGPT defaults all GET operations to false and all other operations to true
    },
)
# /recipes?ingredient=tofu
def get_recipe(ingredient: str):
    docs = vector_store.similarity_search(
        ingredient,
        k=2,
    )
    return docs


user_token_db = {"ABCDEF": "sueweetie"}


@app.get(
    "/authorize",
    response_class=HTMLResponse,
    include_in_schema=False,
)
def handle_authorize(
    client_id: str,
    redirect_uri: str,
    state: str,
):
    return f"""
    <html>
        <head>
            <title>Cutie Sueweetie Log In</title>
        </head>
        <body>
            <h1>Log Into Cutie Sueweetie</h1>
            <a href="{redirect_uri}?code=ABCDEF&state={state}">Authorize Cutie Sueweetie GPT</a>
        </body>
    </html>
    """


@app.post(
    "/token",
    include_in_schema=False,
)
def handle_token(code=Form(...)):
    print(code)
    return {
        "access_token": user_token_db[code],
    }
