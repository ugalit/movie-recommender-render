"""
Contains various recommondation implementations
all algorithms return a list of movieids
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
from utils import movies

def recommend_random(k=5):
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

def recommend_neighborhood(query, ratings, k=5):
    """
    Filters and recommends the top k movies for any given input query
    based on a trained nearest neighbors model.
    Returns a list of k movie ids.
    """

    # 1. candidate generation: movies the user hasn't seen and that have been rated > 50 times
    movies_rated_often = ratings.groupby('movieId')['userId'].count()[
        ratings.groupby('movieId')['userId'].count()>50]
    candidates = ratings[ratings.movieId.isin(movies_rated_often.index)]
    candidates = candidates[~candidates.movieId.isin(list(query.keys()))]
    candidates = candidates.movieId.drop_duplicates().sort_values().to_list()

    # 2. scoring: add user to existing user-user matrix, calculate cosine-similarity
    new_user = pd.DataFrame({'movieId':query.keys(), 'userId':'new_user', 'rating':query.values()})
    ratings_updated = pd.concat([ratings, new_user], join='outer')
    initial = pd.pivot(ratings_updated, index='movieId', columns='userId', values='rating')
    #user_item = initial.fillna(value = 0)
    #user_item = initial.fillna(value = 3.5)
    user_item = initial.astype(np.float).T.fillna(initial.astype(np.float).mean(axis = 1)).T
    user_user = cosine_similarity(user_item.T)
    user_user = pd.DataFrame(
        user_user,
        columns = user_item.columns,
        index = user_item.columns).round(2)
    top_similar = user_user['new_user'].sort_values(ascending=False).index[1:21] # top most similar

    # 3. rating and ranking: top k rating predictions based on weighted average
    # ratings of similar users: sum(ratings*similarity)/sum(similarities)
    movie_predictions = []
    for movie in candidates:
        other_users = initial.columns[~initial.loc[movie].isna()]
        other_users = set(other_users)

        num = 0
        den = 0
        for user in set(top_similar).intersection(other_users):
            ratings = initial[user][movie]      # extract relevant ratings
            sim = user_user['new_user'][user]   # extract relevant cosine sim values
            num = num + (ratings*sim)           # account for "level of similarity"
            den = den + sim

        pred_ratings = num/(den + 0.0000001)
        movie_predictions.append(pred_ratings)

    movie_predictions = pd.DataFrame(
        set(zip(candidates, movie_predictions)),
        columns=['movieId', 'pred_rating'])
    movie_predictions = movie_predictions.sort_values(by='pred_rating', ascending=False)[0:k]
    movie_predictions_out = pd.merge(movie_predictions, movies, how='left', on='movieId').title

    return movie_predictions_out

def recommend_NMF(query, ratings, k=10):
    """
    NMF Recommender
    INPUT
    - user_vector with shape (1, #number of movies)
    - user_item_matrix
    - trained NMF model

    OUTPUT
    - a list of k movieIds
    """

    # 1. NMF model of ratings merged with new user

    new_user = pd.DataFrame({'movieId':query.keys(), 'userId':'new_user', 'rating':query.values()})
    ratings_updated = pd.concat([ratings, new_user], join='outer')
    ratings_updated = pd.pivot(ratings_updated, index='movieId', columns='userId', values='rating')
    #R = ratings_updated.fillna(value = 0).T
    #R = ratings_updated.fillna(value = 3.5).T
    R = round(ratings_updated.astype(np.float).T.fillna(
        ratings_updated.astype(np.float).mean(axis=1)), 2)

    model = NMF(n_components=35, max_iter=5000, random_state=10)#, init='random'
    model.fit(R)
    Q = model.components_  # movie-genre matrix
    print(model.reconstruction_err_)

    # 2. get vector for new user ratings and apply model
    user = ratings_updated['new_user']
    #user_clean = user.fillna(value = 0).values
    #user_clean = user.fillna(value = 3.5).values
    user_clean = R.loc['new_user']

    user_P = model.transform([user_clean])
    user_R = np.dot(user_P,Q)

    # 3. recommendation
    recommendation = pd.DataFrame({
        'user_input':user,
        'predicted_ratings':user_R[0]},
        index = ratings_updated.index)
    recommendation = recommendation[recommendation['user_input'].isna()].sort_values(
        by = 'predicted_ratings', ascending= False)

    return recommendation[:k].index
