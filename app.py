import streamlit as st
import pandas as pd

from backend.profiles import (
    load_user_profiles,
    save_user_profiles,
    generate_new_user_id,
    create_new_profile,
)
from backend.mood import mood_to_genres
from backend.recommender import hybrid_recommend
from backend.data_loader import load_ratings_safe
from backend.config import RATINGS_FILE

st.title("Mood‑Based Hybrid Book Recommender")



# ---------- SESSION INITIALIZATION ----------
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "final_genres" not in st.session_state:
    st.session_state.final_genres = mood_to_genres("neutral")
if "recs" not in st.session_state:
    st.session_state.recs = None

# ---------- 1) LOGIN / SIGNUP ----------
st.header("1) Login / Signup")



profiles = load_user_profiles()
existing_ids = list(profiles.keys())

mode = st.radio("Choose:", ["New user", "Existing user"])

if mode == "New user":
    if st.button("Generate New User ID"):
        new_id = generate_new_user_id()
        st.session_state.user_id = new_id
        st.success(f"Your User ID: {new_id}")

    if st.session_state.user_id:
        st.subheader("Fill your details")
        age = st.text_input("Age")
        gender = st.selectbox("Gender", ["", "M", "F", "Other"])
        country = st.text_input("Country")
        profession = st.text_input("Profession")

        if st.button("Save Profile"):
            profiles = load_user_profiles()
            uid = st.session_state.user_id
            profiles[uid] = create_new_profile(uid, age, gender, country, profession)
            save_user_profiles(profiles)
            st.success("Profile saved!")

else:  # Existing user
    if existing_ids:
        chosen = st.selectbox("Select your User ID", existing_ids)
        if st.button("Login"):
            st.session_state.user_id = chosen
            st.success(f"Logged in as {chosen}")
    else:
        st.info("No existing users. Please create a new user first.")

uid = st.session_state.user_id

# ---------- 2) MOOD INPUT ----------
st.header("2) Mood input")

if not uid:
    st.warning("Please login or create a user first.")
else:
    choice = st.radio(
        "How do you want to give your mood?",
        ["Text choice", "Selfie (upload only)"]
    )

    # just for display; real genres live in st.session_state.final_genres
    final_mood = "neutral"

    # ----- TEXT CHOICE -----
    if choice == "Text choice":
        mood_choice = st.selectbox(
            "Select your current mood:",
            ["joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral"],
        )
        if st.button("Use this mood"):
            final_mood = mood_choice
            st.session_state.final_genres = mood_to_genres(mood_choice)
            st.write(f"Mood: {final_mood}")
            st.write(f"Genres: {st.session_state.final_genres}")

    # ----- SELFIE UPLOAD (NO DEEPFACE) -----
    else:
        img = st.file_uploader("Upload a selfie image", type=["jpg", "jpeg", "png"])
        if img is not None:
            st.image(img, caption="Your selfie", use_column_width=True)
            st.info("In the full system, this selfie would be analyzed with DeepFace.")

        mood_choice2 = st.selectbox(
            "Select mood that matches your selfie:",
            ["joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral"],
            key="selfie_mood",
        )
        if st.button("Use selfie mood"):
            final_mood = mood_choice2
            st.session_state.final_genres = mood_to_genres(mood_choice2)
            st.write(f"Mood from selfie: {final_mood}")
            st.write(f"Genres: {st.session_state.final_genres}")

    # ---------- 3) HYBRID RECOMMENDATIONS ----------
    st.header("3) Hybrid recommendations")

    # If user clicks Recommend, compute and store recs
    if st.button("Recommend Books"):
        recs = hybrid_recommend(uid, st.session_state.final_genres, alpha=0.6, top_n=5)
        st.session_state.recs = recs

    # If we have stored recs, show them and rating inputs
    if st.session_state.recs is not None:
        recs = st.session_state.recs
        st.dataframe(recs)

        st.subheader("Rate these books (optional)")
        ratings = load_ratings_safe()
        new_rows = []

        for i, row in recs.iterrows():
            isbn = row["isbn"]
            title = row["title"]
            rating = st.number_input(
                f"Your rating for '{title}' (1–10, or 0 to skip):",
                min_value=0, max_value=10, step=1, key=f"rate_{isbn}"
            )
            if rating > 0:
                new_rows.append({"User-ID": uid, "ISBN": isbn, "Rating": rating})

        if st.button("Save Ratings"):
            if new_rows:
                new_df = pd.DataFrame(new_rows)
                ratings = pd.concat([ratings, new_df], ignore_index=True)
                ratings.drop_duplicates(subset=["User-ID", "ISBN"], keep="last", inplace=True)
                ratings.to_csv(RATINGS_FILE, sep=";", index=False)
                st.success("Ratings saved and will be used in future recommendations.")
            else:
                st.info("No ratings provided.")
