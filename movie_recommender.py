"""
Content-Based Movie Recommender
================================
Recommends similar movies based on content (genres, cast, keywords, overview)
using TF-IDF vectorization + cosine similarity.

This is a real, deployable recommendation approach used by streaming services:
it converts movie metadata into vectors and measures how "close" movies are.

Libraries: scikit-learn, pandas
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# A curated movie dataset (title, genres, cast, director, keywords, overview).
# In production you would load a larger CSV here instead (see load_data).
MOVIES = [
    ("The Dark Knight", "Action Crime Drama", "Christian Bale Heath Ledger Aaron Eckhart", "Christopher Nolan",
     "superhero batman gotham villain chaos", "Batman must face the Joker, a criminal mastermind who plunges Gotham into anarchy."),
    ("Inception", "Action Sci-Fi Thriller", "Leonardo DiCaprio Joseph Gordon-Levitt Tom Hardy", "Christopher Nolan",
     "dream heist mind manipulation subconscious", "A thief who steals corporate secrets through dream-sharing technology is given the task of planting an idea."),
    ("Interstellar", "Adventure Drama Sci-Fi", "Matthew McConaughey Anne Hathaway Jessica Chastain", "Christopher Nolan",
     "space time travel black hole exploration", "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival."),
    ("The Matrix", "Action Sci-Fi", "Keanu Reeves Laurence Fishburne Carrie-Anne Moss", "The Wachowskis",
     "simulation reality hacker machines", "A hacker learns the shocking truth about his reality and his role in the war against its controllers."),
    ("Avengers: Endgame", "Action Adventure Sci-Fi", "Robert Downey Jr Chris Evans Mark Ruffalo", "Anthony Russo Joe Russo",
     "superhero marvel time travel infinity stones", "The Avengers assemble once more to reverse Thanos' actions and restore balance to the universe."),
    ("The Godfather", "Crime Drama", "Marlon Brando Al Pacino James Caan", "Francis Ford Coppola",
     "mafia family crime power", "The aging patriarch of an organized crime dynasty transfers control of his empire to his reluctant son."),
    ("Pulp Fiction", "Crime Drama", "John Travolta Samuel L. Jackson Uma Thurman", "Quentin Tarantino",
     "nonlinear crime hitman underworld", "The lives of two mob hitmen, a boxer, a gangster and his wife intertwine in four tales of violence and redemption."),
    ("The Shawshank Redemption", "Drama", "Tim Robbins Morgan Freeman Bob Gunton", "Frank Darabont",
     "prison friendship hope escape", "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency."),
    ("Forrest Gump", "Drama Romance", "Tom Hanks Robin Wright Gary Sinise", "Robert Zemeckis",
     "history love destiny journey", "The presidencies of Kennedy and Johnson, the events of Vietnam, and other history unfold through the perspective of an Alabama man."),
    ("Fight Club", "Drama Thriller", "Brad Pitt Edward Norton Helena Bonham Carter", "David Fincher",
     "psychology identity underground society", "An insomniac office worker forms an underground fight club that evolves into something much more."),
    ("The Silence of the Lambs", "Crime Drama Thriller", "Jodie Foster Anthony Hopkins Scott Glenn", "Jonathan Demme",
     "serial killer fbi psychological suspense", "A young FBI cadet must receive the help of an incarcerated and manipulative cannibal killer to catch another serial killer."),
    ("Spirited Away", "Animation Family Fantasy", "Rumi Hiiragi Miyu Irino Mari Natsuki", "Hayao Miyazaki",
     "spirits anime adventure coming of age", "A young girl wanders into a world ruled by gods, witches, and spirits, where humans are changed into beasts."),
    ("Toy Story", "Animation Adventure Comedy", "Tom Hanks Tim Allen Don Rickles", "John Lasseter",
     "toys friendship childhood adventure", "A cowboy doll is profoundly threatened and jealous when a new spaceman figure supplants him as top toy in a boy's room."),
    ("The Lion King", "Animation Drama Family", "Matthew Broderick Jeremy Irons James Earl Jones", "Roger Allers Rob Minkoff",
     "africa pride circle of life lion", "Lion prince Simba and his father are targeted by his bitter uncle who wants to ascend the throne himself."),
    ("Jaws", "Adventure Thriller", "Roy Scheider Robert Shaw Richard Dreyfuss", "Steven Spielberg",
     "shark ocean horror survival", "A police chief, a marine biologist, and a grizzled shark hunter search for a killer great white shark."),
    ("Jurassic Park", "Adventure Sci-Fi Thriller", "Sam Neill Laura Dern Jeff Goldblum", "Steven Spielberg",
     "dinosaurs park science cloning", "During a preview tour, a theme park suffers a major power breakdown that allows its cloned dinosaur exhibits to run amok."),
    ("Titanic", "Drama Romance", "Leonardo DiCaprio Kate Winslet Billy Zane", "James Cameron",
     "ship disaster love iceberg", "A seventeen-year-old aristocrat falls in love with a kind but poor artist aboard the luxurious, ill-fated R.M.S. Titanic."),
    ("Gladiator", "Action Adventure Drama", "Russell Crowe Joaquin Phoenix Connie Nielsen", "Ridley Scott",
     "rome gladiator revenge empire", "A former Roman General sets out to exact vengeance against the corrupt emperor who murdered his family."),
    ("The Departed", "Crime Drama Thriller", "Leonardo DiCaprio Matt Damon Jack Nicholson", "Martin Scorsese",
     "undercover police mafia deception", "An undercover cop and a mole in the police attempt to identify each other while infiltrating an Irish gang."),
    ("Whiplash", "Drama Music", "Miles Teller J.K. Simmons Paul Reiser", "Damien Chazelle",
     "jazz drummer ambition mentor", "A promising young drummer enrolls at a cut-throat music conservatory where his dreams of greatness are mentored by an instructor who will stop at nothing to realize a student's potential."),
    ("Mad Max: Fury Road", "Action Adventure Sci-Fi", "Tom Hardy Charlize Theron Nicholas Hoult", "George Miller",
     "post-apocalyptic chase desert cars", "In a post-apocalyptic wasteland, a woman rebels against a tyrannical ruler in search of her homeland."),
    ("Get Out", "Horror Mystery Thriller", "Daniel Kaluuya Allison Williams Bradley Whitford", "Jordan Peele",
     "social horror racism suspense", "A young African-American visits his white girlfriend's parents for the weekend, where his simmering uneasiness about their reception of him eventually reaches a boiling point."),
    ("The Social Network", "Biography Drama", "Jesse Eisenberg Andrew Garfield Justin Timberlake", "David Fincher",
     "facebook startup technology law", "As Harvard student Mark Zuckerberg creates the social networking site that would become known as Facebook, he is sued by the twins who claimed he stole their idea."),
    ("La La Land", "Comedy Drama Music", "Ryan Gosling Emma Stone John Legend", "Damien Chazelle",
     "love hollywood dreams jazz", "While navigating their careers in Los Angeles, a pianist and an actress fall in love while attempting to reconcile their aspirations for the future."),
    ("The Grand Budapest Hotel", "Adventure Comedy", "Ralph Fiennes F. Murray Abraham Mathieu Amalric", "Wes Anderson",
     "hotel comedy heist quirky", "A writer encounters the owner of an aging high-class hotel, who tells him of his early years as a lobby boy in the hotel's glorious years under an exceptional concierge."),
    ("Parasite", "Comedy Drama Thriller", "Song Kang-ho Lee Sun-kyun Cho Yeo-jeong", "Bong Joon-ho",
     "class satire family poverty social", "Greed and class discrimination threaten the newly formed symbiotic relationship between the wealthy Park family and the destitute Kim clan."),
    ("Dunkirk", "Action Drama History", "Fionn Whitehead Barry Keoghan Mark Rylance", "Christopher Nolan",
     "war world war two survival evacuation", "Allied soldiers from Belgium, the British Empire, and France are surrounded by the German army and evacuated during a fierce battle in World War II."),
]


def build_features():
    """Combine all metadata into one searchable 'document' per movie."""
    docs = []
    for title, genres, cast, director, keywords, overview in MOVIES:
        # Combine all fields; TF-IDF will weight the meaningful words
        text = " ".join([genres, cast, director, keywords, overview]).lower()
        docs.append(text)
    return docs


def load_data():
    """Try to load an optional larger CSV; fall back to the built-in list."""
    try:
        df = pd.read_csv("movies.csv")
        if len(df) > len(MOVIES):
            print(f"Loaded {len(df)} movies from movies.csv")
            df["text"] = df[["genres", "cast", "director", "keywords", "overview"]].fillna("").agg(
                " ".join, axis=1).str.lower()
            return df
    except FileNotFoundError:
        pass
    print(f"Using built-in dataset ({len(MOVIES)} movies)")
    return pd.DataFrame(
        MOVIES, columns=["title", "genres", "cast", "director", "keywords", "overview"]
    ).assign(text=build_features())


def recommend(df, title, top_n=5):
    """Return the top_n most similar movies to the given title."""
    # Build TF-IDF vectors from combined text
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(df["text"])

    # Cosine similarity: how similar is every movie to every other movie
    similarity = cosine_similarity(tfidf)

    # Find the index of the given movie
    titles = df["title"].str.lower()
    try:
        idx = list(titles).index(title.lower())
    except ValueError:
        return f"Movie '{title}' not found. Try one of: {list(df['title'][:10])}"

    # Sort movies by similarity (descending), skip the movie itself
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1 : top_n + 1]

    print(f"\nBecause you watched: {df['title'][idx]}")
    print("=" * 55)
    for i, score in sim_scores:
        print(f"  {score:.3f}  {df['title'][i]}")


def main():
    df = load_data()
    print("\nBuilding TF-IDF vectors and similarity matrix...")

    for movie in ["The Dark Knight", "Inception", "Toy Story", "Parasite", "The Godfather"]:
        recommend(df, movie)


if __name__ == "__main__":
    main()
