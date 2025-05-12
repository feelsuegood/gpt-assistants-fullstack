from http import server
from fastapi import FastAPI
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
)
def get_treat():
    return {
        "treat": "Action speaks louder than words.",
        "level": 99,
    }
