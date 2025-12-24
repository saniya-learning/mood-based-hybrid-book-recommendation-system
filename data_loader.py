import pandas as pd
from .config import RATINGS_FILE, BOOKS_FILE

def load_ratings_safe(sample_size=10000):
    df = pd.read_csv(RATINGS_FILE, sep=";", encoding="latin-1", on_bad_lines="skip")
    df.columns = df.columns.str.strip().str.lower().str.replace("-", "_")
    # handle both book_rating (original) and rating (new)
    if "book_rating" in df.columns:
        r = df[["user_id", "isbn", "book_rating"]].copy()
    elif "rating" in df.columns:
        r = df[["user_id", "isbn", "rating"]].copy()
    else:
        raise KeyError("No rating or book_rating column in Ratings file")

    r.columns = ["User-ID", "ISBN", "Rating"]
    r["User-ID"] = r["User-ID"].astype(str).str.strip()
    r["ISBN"] = r["ISBN"].astype(str).str.strip()
    r["Rating"] = pd.to_numeric(r["Rating"], errors="coerce").fillna(0).clip(0, 10)
    return r if len(r) <= sample_size else r.sample(sample_size, random_state=42)

def load_books_safe():
    df = pd.read_csv(BOOKS_FILE, sep=";", encoding="latin-1", on_bad_lines="skip")
    df.columns = df.columns.str.strip().str.lower().str.replace("-", "_")
    b = df[["isbn", "book_title", "book_author", "publisher"]].copy()
    b.rename(columns={"book_title": "title", "book_author": "author"}, inplace=True)
    b["isbn"] = b["isbn"].astype(str).str.strip()
    b["publisher"] = b["publisher"].astype(str).str.strip()
    return b
