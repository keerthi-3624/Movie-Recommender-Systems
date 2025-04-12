# Movie Recommendation System Project

## Project Overview

This project implements a **Movie Recommendation System** using two primary approaches:
1. **Content-Based Filtering**: Recommends movies similar to a given movie based on their descriptions (overviews).
2. **Collaborative Filtering**: Recommends movies to a user based on the preferences of other users with similar tastes, using matrix factorization techniques.

The system is built using Python and popular libraries such as `pandas`, `numpy`, `scikit-learn`, and `scipy`.

## Features of the Dataset

The project uses two datasets:
1. **movies_metadata.csv**: Contains metadata about movies, including their titles and overviews.
2. **ratings_small.csv**: Contains user ratings for various movies.

### Key Columns in the Datasets:
- **movies_metadata.csv**:
  - `id`: Unique identifier for each movie.
  - `title`: Title of the movie.
  - `overview`: A brief description of the movie's plot.

- **ratings_small.csv**:
  - `userId`: Unique identifier for each user.
  - `movieId`: Unique identifier for each movie rated by users.
  - `rating`: The rating given by a user to a movie (on a scale of 0-5).

## Libraries Used

The following Python libraries are used in this project:
- **pandas**: For data manipulation and handling DataFrames.
- **numpy**: For numerical operations.
- **scikit-learn**:
  - `TfidfVectorizer`: Converts movie overviews into numerical feature vectors.
  - `cosine_similarity`: Computes similarity between movies based on their feature vectors.
- **scipy.sparse.linalg**:
  - `svds`: Performs Singular Value Decomposition (SVD) for collaborative filtering.

## Workflow

### 1. Data Loading and Preprocessing
- Load the datasets (`movies_metadata.csv` and `ratings_small.csv`) into pandas DataFrames.
- Select relevant columns (`id`, `title`, and `overview`) from the movies dataset and remove missing values.
- Convert the `id` column to numeric format and reset the index.

### 2. Content-Based Filtering
- Use **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert movie overviews into numerical representations.
- Compute pairwise cosine similarity between all movies based on their TF-IDF vectors.
- Define a function to recommend movies similar to a given movie based on cosine similarity scores.

### 3. Collaborative Filtering
- Create a user-item rating matrix from the ratings dataset, where rows represent users, columns represent movies, and values represent ratings.
- Perform **Singular Value Decomposition (SVD)** to reduce the dimensionality of the rating matrix while capturing important patterns in user preferences.
- Reconstruct the predicted rating matrix from the SVD components.
- Define a function to recommend movies to a user based on predicted ratings.

### 4. Recommendation Functions
Two main functions are provided:
1. **recommend_content(title, num_recommendations=5)**: Recommends movies similar to the given title using content-based filtering.
2. **recommend_collaborative(user_id, num_recommendations=5)**: Recommends movies for a specific user using collaborative filtering.

## How to Run the Code

1. Clone this repository to your local machine.
2. Ensure that Python is installed along with required libraries (`pandas`, `numpy`, `scikit-learn`, `scipy`).
3. Place the datasets (`movies_metadata.csv` and `ratings_small.csv`) in the working directory.
4. Open the Jupyter Notebook file (`Movie_recomm.ipynb`) in Jupyter Notebook or any compatible IDE.
5. Run all cells sequentially to execute the code.

### Example Usage
```python
# Content-based recommendations
print(recommend_content("The Godfather"))

# Collaborative filtering recommendations for user ID 1
print(recommend_collaborative(1))
```

## Results

The system provides personalized recommendations for users or suggests similar movies based on content. Example outputs include:
- Content-based recommendations for "The Godfather": ['The Godfather: Part II', 'The Godfather Trilogy', ...]
- Collaborative filtering recommendations for User ID 1: [1374, 2968, ...]

## Future Improvements
- Incorporate additional features like genres, cast, or release year for better recommendations.
- Experiment with deep learning models like autoencoders for collaborative filtering.
- Optimize hyperparameters and increase scalability for larger datasets.
