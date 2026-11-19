# movie-api/main.py
import os

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from model import model

app = FastAPI()

CSV_PATH = os.path.join(os.path.dirname(__file__), "movie_dataset.csv")


class RecommendRequest(BaseModel):
    title: str
    count: int = 5


@app.get("/titles")
def titles():
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    return {"titles": df["title"].tolist()}


@app.post("/recommend")
def get_recommendations(body: RecommendRequest):
    return {"films": model(body.title, body.count)}
