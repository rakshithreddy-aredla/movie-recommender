# Content-based movie recommender

Given a film, finds films that are similar on content — genres, cast, director, keywords, plot — using TF-IDF vectors and cosine similarity.

```
Because you watched: The Dark Knight
  0.067  The Godfather
  0.059  Inception
  0.058  Dunkirk
  0.051  Interstellar
  0.048  Pulp Fiction
```

Each movie's text fields are concatenated into one document, vectorized, then compared against every other film. Cosine similarity measures the angle between vectors, which is why it works here: what separates two films is direction (what they're about), not magnitude (how much text they have).

The similarity scores are low in absolute terms — 0.067 is a good match for this kind of data. Plot summaries share so much vocabulary that vectors cluster tightly, and the ranking matters more than the raw value. It's a content-based recommender, so it will happily recommend other Christopher Nolan films before it finds anything genuinely different.

```bash
pip install -r requirements.txt
python movie_recommender.py
```

Drop a `movies.csv` with `title, genres, cast, director, keywords, overview` in the folder and the script picks it up automatically.

## Files

```
movie_recommender.py   # vectorize, compare, rank
requirements.txt
```