import os

import pandas as pd
import requests
from decouple import config
from dotenv import load_dotenv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_BASE_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.dirname(_BASE_DIR)

# Prefer movie_model/.env, then sibling movieaisite/.env for local monorepo use.
load_dotenv(dotenv_path=os.path.join(_BASE_DIR, ".env"))
load_dotenv(dotenv_path=os.path.join(_REPO_ROOT, "movieaisite", ".env"))

CSV_PATH = os.path.join(_BASE_DIR, "movie_dataset.csv")


def model(name, num):
    df = pd.read_csv(CSV_PATH, encoding="utf-8")

    features = ["keywords", "cast", "genres", "director"]

    def combine_features(row):
        return row["keywords"] + " " + row["cast"] + " " + row["genres"] + " " + row["director"]

    for feature in features:
        df[feature] = df[feature].fillna("")

    df["combined_features"] = df.apply(combine_features, axis=1)

    cv = CountVectorizer()
    count_matrix = cv.fit_transform(df["combined_features"])

    cosine_sim = cosine_similarity(count_matrix)

    def get_index_from_title(title):
        return df[df.title == title]["index"].values[0]

    def get_data_from_index(index):
        return [
            df[df.index == index]["title"].values[0],
            df[df.index == index]["genres"].values[0],
            df[df.index == index]["director"].values[0],
            df[df.index == index]["cast"].values[0],
            df[df.index == index]["overview"].values[0],
            getposter(df[df.index == index]["title"].values[0]),
        ]

    movie_index = get_index_from_title(name)
    similar_movies = list(enumerate(cosine_sim[movie_index]))
    sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:]

    context = []
    i = 0
    for element in sorted_similar_movies:
        context.append(get_data_from_index(element[0]))
        i = i + 1
        if i > num:
            break
    return context


def authorizeTMDB():
    url = "https://api.themoviedb.org/3/authentication"
    headers = {
        "accept": "application/json",
        "Authorization": "Bearer " + config("MOVIE_API_KEY"),
    }
    return requests.get(url, headers=headers)


def getposter(title):
    api_key = config("MOVIE_API_KEY", default=None)
    if not api_key:
        return None

    url = "https://api.themoviedb.org/3/search/movie"
    headers = {
        "accept": "application/json",
        "Authorization": "Bearer " + api_key,
    }
    params = {
        "query": title,
        "include_adult": "false",
        "language": "en-US",
        "page": 1,
    }
    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
    except requests.RequestException:
        return None

    if response.status_code != 200:
        return None

    data = response.json()
    results = data.get("results") or []
    if not results:
        return None

    poster_path = results[0].get("poster_path")
    if not poster_path:
        return None

    return f"https://image.tmdb.org/t/p/w500{poster_path}"
