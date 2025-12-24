# mood-based-hybrid-book-recommendation-system
Hybrid book recommender that combines mood detection with content-based and collaborative filtering for better recommendations. Streamlit is used for UI.
## Project overview

Mood-Based Hybrid Book Recommender that combines mood detection with content-based and collaborative filtering to generate personalized book suggestions.

## How to run

1. Install Python dependencies:
2. Start the Streamlit app from this folder:

## Features

- User login / signup with basic profile
- Mood input (text choice or selfie + mood dropdown)
- Hybrid recommender: collaborative filtering + content-based filtering (TF-IDF on book metadata)
- Users can rate recommended books; ratings are saved to `Ratings.csv`

## Implementation notes

- In Google Colab, the project includes transformer-based text mood detection and DeepFace-based emotion recognition from selfies.
- In the local Streamlit app (Windows, Python 3.12), mood is selected via dropdown because TensorFlow / DeepFace could not be installed reliably.
- Evaluation includes a global-mean RMSE baseline on a 20k-rating subset of the Book-Crossing dataset.
