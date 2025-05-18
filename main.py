from typing import Any
from fastapi import Body, FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

app = FastAPI(
    title="Cutie Sueweetie Treats Giver",
    description="Get a love magic treats given by Cutie Sueweetie herself.",
    servers=[
        {
            "url": "https://immigrants-pray-albania-lips.trycloudflare.com",
        },
    ],
)


class Treat(BaseModel):
    treat: str = Field(description="The love magic treat that Cutie Sweetie gave.")
    level: int = Field(
        description="The level of treat that represents how powerful it is."
    )


@app.get(
    "/treat",
    summary="Returns a random treat by Cutie Sueweetie",
    description="Upon receiving a GET request this endpoint will return a love magic treat given by Cutie Sueweetie herself.",
    response_description="A treat object that contains the treat given by Cutie Sueweetie and the level that the love magic treat has from 0 to 100.",
    response_model=Treat,
    openapi_extra={
        "x-openai-isConsequential": True,
        # If the x-openai-isConsequential field is true, ChatGPT treats the operation as "must always prompt the user for confirmation before running" and don't show an "always allow" button (both are features of GPTs designed to give builders and users more control over actions).
        # If the x-openai-isConsequential field is false, ChatGPT shows the "always allow button".
        # If the field isn't present, ChatGPT defaults all GET operations to false and all other operations to true
    },
)
def get_treat(request: Request):
    print(request.headers)
    return {
        "treat": "Action speaks louder than words.",
        "level": 99,
    }


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
