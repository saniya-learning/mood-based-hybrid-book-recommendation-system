import pickle, numpy as np, pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from .config import TFIDF_VECTORIZER_FILE, TFIDF_MATRIX_FILE, USER_PROFILES_FILE
from .data_loader import load_ratings_safe, load_books_safe
from .profiles import load_user_profiles, save_user_profiles

with open(TFIDF_VECTORIZER_FILE,"rb") as f:
    V = pickle.load(f)
with open(TFIDF_MATRIX_FILE,"rb") as f:
    M = pickle.load(f)

def hybrid_recommend(user_id, genres, alpha=0.6, top_n=5):
    ratings = load_ratings_safe()
    books = load_books_safe()

    # CF
    ui = ratings.pivot_table(index="User-ID", columns="ISBN", values="Rating", aggfunc="mean").fillna(0)
    means = ui.replace(0, np.nan).mean(axis=1).fillna(0)
    ctr = ui.sub(means, axis=0).fillna(0)
    if user_id not in ctr.index:
        popular = ratings["ISBN"].value_counts().head(top_n).index
        cf = books[books["isbn"].isin(popular)][["isbn","title","author"]].copy()
    else:
        sims = cosine_similarity(ctr.loc[[user_id]], ctr)[0]
        s = pd.Series(sims, index=ctr.index).drop(user_id).sort_values(ascending=False).head(10)
        scores = ctr.loc[s.index].T.dot(s) / s.sum()
        preds = means[user_id] + scores
        top = preds.sort_values(ascending=False).head(top_n).index
        cf = books[books["isbn"].isin(top)][["isbn","title","author"]].copy()
    cf["source"] = "CF"; cf["score"] = alpha

    # TF-IDF from genres
    if not genres:
        genres = ["General Fiction", "Drama"]
    q = " ".join(genres)
    sims = cosine_similarity(V.transform([q]), M).flatten()
    idx = sims.argsort()[-top_n:][::-1]
    tf = books.iloc[idx][["isbn","title","author"]].copy()
    tf["source"] = "TF-IDF"; tf["score"] = 1 - alpha

    hybrid = pd.concat([cf, tf]).drop_duplicates("isbn")
    hybrid = hybrid.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)

    profiles = load_user_profiles()
    if user_id in profiles:
        profiles[user_id]["last_recommendations"] = hybrid.to_dict(orient="records")
        save_user_profiles(profiles)

        return hybrid[["isbn", "title", "author"]]

