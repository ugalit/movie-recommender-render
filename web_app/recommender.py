"""
Contains various recommondation implementations
all algorithms return a list of movieids
"""

import pandas as pd
import numpy as np
from utils import movies

def recommend_random(k=3):
    """
    Return random movies.
    """
    return movies['title'].sample(k)

def recommend_popular(query, ratings, k=5):
    """
    Filters and recommends the top k movies for any given input query. 
    """
    # 1. candidate generation: filter out movies already seen and watched by < 50 users
    movies_rated_often = ratings.groupby('movieId')['userId'].count()[
        ratings.groupby('movieId')['userId'].count()>50]
    candidates = ratings[ratings.movieId.isin(movies_rated_often.index)]
    candidates = candidates[~candidates.movieId.isin(list(query.keys()))]
    
    # 2. scoring: average rating for each movie
    scores = candidates.groupby('movieId')['rating'].mean()
    
    # 3. ranking: top-k highest rated movie ids or titles
    best_k = scores.sort_values(ascending=False).head(k).index

    return list(best_k)

def recommend_with_NMF(query, k=3):
    """
    NMF Recommender
    INPUT
    - user_vector with shape (1, #number of movies)
    - user_item_matrix
    - trained NMF model

    OUTPUT
    - a list of movieIds
    """
    pass

def recommend_neighborhood(query, model, k=3):
    """
    Filters and recommends the top k movies for any given input query based on a trained nearest neighbors model. 
    Returns a list of k movie ids.
    """   
    pass
