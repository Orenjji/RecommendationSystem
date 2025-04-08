Here’s a `README.md` for your Movie Recommender System using the Universal Sentence Encoder and Nearest Neighbors:

---

# Movie Recommender System using Machine Learning and NLP

This Movie Recommender System leverages Natural Language Processing (NLP) and machine learning techniques to recommend movies based on textual input. The system uses the Universal Sentence Encoder to embed movie descriptions and then uses the Nearest Neighbors algorithm to find the most similar movies.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Getting Started](#getting-started)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Contributing](#contributing)
- [License](#license)

## Overview

This Movie Recommender System suggests movies to users based on their preferences expressed as text input. It works by embedding movie descriptions into vector space using the Universal Sentence Encoder (a pre-trained NLP model). The system then uses the Nearest Neighbors algorithm to find movies that are most similar to the input, taking into account movie popularity and vote averages for sorting.

## Features

- **Text-Based Movie Recommendations**: Provide input text such as preferences (e.g., "I like thrillers but not robberies") and receive movie suggestions based on similarity to movie descriptions.
- **Cosine Similarity**: Uses cosine similarity between text embeddings to find the most similar movies.
- **Popularity and Ratings**: Sorts recommended movies based on popularity and user ratings.

## Getting Started

To run the Movie Recommender System, follow these steps to set up the environment and begin generating movie recommendations.

### Prerequisites

- **Python 3.x**: This project is built with Python 3.
- **Libraries**: You'll need to install several Python libraries to run the code.

### Installation

1. **Clone the Repository**:
   Clone this repository to your local machine or use it in a cloud environment like Google Colab:
   ```bash
   git clone https://github.com/your-username/movie-recommender.git
   ```

2. **Install Required Libraries**:
   Install the necessary Python libraries using `pip`:
   ```bash
   pip install tensorflow tensorflow-hub sklearn pandas numpy matplotlib sentencepiece
   ```

3. **Download the Dataset**:
   Make sure to have a dataset similar to the `Top_10000_Movies.csv` file. This dataset should contain movie titles, descriptions, genres, popularity scores, and vote averages.

## Usage

1. **Load the Model**:
   First, the Universal Sentence Encoder model is loaded to convert movie descriptions into embeddings.
   ```python
   import tensorflow_hub as hub
   model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
   model = hub.load(model_url)
   ```

2. **Preprocess Data**:
   Load your dataset (in CSV format) containing movie information, such as titles, descriptions, popularity, and ratings:
   ```python
   import pandas as pd
   df = pd.read_csv("Top_10000_Movies.csv", engine="python")
   df = df[["original_title", "overview", "genre", "popularity", "vote_average"]]
   df = df.dropna().reset_index()
   ```

3. **Embed Text**:
   Use the Universal Sentence Encoder to convert movie descriptions into vectors:
   ```python
   def embed(texts):
       return model(texts)
   ```

4. **Fit Nearest Neighbors Model**:
   After embedding the movie descriptions, use the Nearest Neighbors algorithm to find similar movies:
   ```python
   from sklearn.neighbors import NearestNeighbors
   embeddings = embed(df['overview'].values)
   nn = NearestNeighbors(n_neighbors=6, metric='cosine')
   nn.fit(embeddings)
   ```

5. **Get Movie Recommendations**:
   Define a function to get movie recommendations based on a given input text:
   ```python
   def recommend(text):
       emb = embed([text])  # Embed the input text
       neighbors = nn.kneighbors(emb, return_distance=False)[0]  # Find the nearest neighbors
       recommended_movies = df.iloc[neighbors]  # Get the corresponding movie details
       recommended_movies = recommended_movies.sort_values(by=["popularity", "vote_average"], ascending=False)
       return recommended_movies[["original_title", "popularity", "vote_average"]].values.tolist()
   ```

6. **Test the Recommendation System**:
   Input your preferences to receive movie recommendations:
   ```python
   recommendations = recommend('i dont like robbery but i like thriller')
   for movie in recommendations:
       print(f'Title: {movie[0]},\t \t Popularity: {movie[1]}, \t \t Vote Average: {movie[2]}')
   ```

## Model

- **Universal Sentence Encoder**: A pre-trained NLP model from TensorFlow Hub used to embed movie descriptions into vector space.
- **Nearest Neighbors**: A machine learning algorithm used to find the nearest neighbors based on the cosine similarity of the movie embeddings.

## Contributing

If you'd like to contribute to this project, feel free to fork the repository and create a pull request. Here’s how you can contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit the changes (`git commit -am 'Add a new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
