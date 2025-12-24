mood_genre_map = {
    "joy":      ["Comedy", "Romance", "Adventure"],
    "sadness":  ["Comedy", "Self-Help"],
    "anger":    ["Comedy", "Romance"],
    "fear":     ["Fantasy", "Comedy","Romance"],
    "disgust":  ["Self-Help", "Comedy"],
    "surprise": ["Comedy", "Adventure", "Fantasy"],
    "neutral":  ["General Fiction", "Classics", "Romance"],
}

def mood_to_genres(mood: str):
    return mood_genre_map.get(mood.lower(), mood_genre_map["neutral"])
