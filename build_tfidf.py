import re, pickle, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from .config import BOOKS_FILE, TFIDF_VECTORIZER_FILE, TFIDF_MATRIX_FILE

books = pd.read_csv(BOOKS_FILE, sep=";", encoding="latin-1", on_bad_lines="skip")
books.columns = books.columns.str.strip().str.lower().str.replace("-", "_")
books = books[["isbn","book_title","book_author","publisher"]].dropna(subset=["book_title","book_author"])

for col in ["isbn","book_title","book_author","publisher"]:
    books[col] = books[col].astype(str).str.strip()

books.reset_index(drop=True, inplace=True)

def clean_text(t):
    t = str(t).lower()
    t = re.sub(r"[^a-z\s]", "", t)
    return t.strip()

books["combined_text"] = (
    books["book_title"].apply(clean_text) + " " +
    books["book_author"].apply(clean_text) + " " +
    books["publisher"].apply(clean_text)
)

vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
tfidf_matrix = vectorizer.fit_transform(books["combined_text"])

with open(TFIDF_VECTORIZER_FILE,"wb") as f:
    pickle.dump(vectorizer,f)
with open(TFIDF_MATRIX_FILE,"wb") as f:
    pickle.dump(tfidf_matrix,f)

print("TF‑IDF vectorizer and matrix saved.")
