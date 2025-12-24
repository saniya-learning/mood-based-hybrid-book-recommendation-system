import os
import pickle
from .config import USER_PROFILES_FILE, USER_ID_TRACKER_FILE

def load_user_profiles():
    if os.path.exists(USER_PROFILES_FILE):
        with open(USER_PROFILES_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def save_user_profiles(profiles):
    with open(USER_PROFILES_FILE, "wb") as f:
        pickle.dump(profiles, f)

def generate_new_user_id():
    last_id = 0
    if os.path.exists(USER_ID_TRACKER_FILE):
        with open(USER_ID_TRACKER_FILE, "r") as f:
            txt = f.read().strip()
            if txt.isdigit():
                last_id = int(txt)
    new_id_num = last_id + 1
    with open(USER_ID_TRACKER_FILE, "w") as f:
        f.write(str(new_id_num))
    return f"U{new_id_num:03d}"

def create_new_profile(user_id, age="", gender="", country="", profession=""):
    return {
        "user_id": user_id,
        "age": age,
        "gender": gender,
        "country": country,
        "profession": profession,
        "mood_history": [],
        "genre_preferences": {},
        "rated_books": {},
        "last_recommendations": []
    }
