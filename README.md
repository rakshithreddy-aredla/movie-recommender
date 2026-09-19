# Movie Recommender System 🎬

A **content-based recommendation engine** that finds similar movies using TF-IDF text vectorization and cosine similarity — the same core technique behind streaming service recommendations.

## 🎯 The Problem

Recommendation systems power Netflix, YouTube, Amazon, and Spotify. This project implements the **content-based** approach: given a movie you like, find other movies with similar content (genres, cast, keywords, plot).

## 🔧 How it works

| Step | What happens |
|------|-------------|
| 1. Data | Movie metadata: title, genres, cast, director, keywords, overview |
| 2. Combine | All text fields merged into one "document" per movie |
| 3. Vectorize | **TF-IDF** converts each document into a numeric vector |
| 4. Compare | **Cosine similarity** measures how similar every pair of movies is |
| 5. Recommend | Sort by similarity, return the top-N most similar movies |

## 🧠 Why cosine similarity?

Movies are represented as vectors in a high-dimensional space. Cosine similarity measures the **angle** between vectors, ignoring magnitude — so it captures *direction of interest* (what topics a movie is about) rather than raw word counts.

## 📊 Example Output

```
Because you watched: The Dark Knight
=======================================================
  0.067  The Godfather
  0.059  Inception
  0.058  Dunkirk
  0.051  Interstellar
  0.048  Pulp Fiction
```

The recommendations make intuitive sense — Dark Knight fans get other acclaimed action/drama/crime films.

## 🚀 How to run

```bash
pip install -r requirements.txt
python movie_recommender.py
```

> **Note:** To use a larger dataset, drop a `movies.csv` with columns
> `title, genres, cast, director, keywords, overview` into this folder.
> The script automatically uses it if present.

## 🏗️ Project Structure

```
04-movie-recommender/
├── movie_recommender.py  # Main script
├── requirements.txt
└── README.md
```

## 📚 ML Concepts Covered

- Recommendation systems (content-based filtering)
- TF-IDF text vectorization
- Cosine similarity for measuring vector closeness
- Building features from unstructured text metadata
- The design decision of combining multiple text fields

## 💡 To Extend (great hackathon ideas)

- Add **collaborative filtering** (user ratings) to combine with content-based
- Build a **Flask web UI** so users search movies in the browser
- Add a popularity/rating hybrid weighting
