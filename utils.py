"""
UTILS
- Helper functions to use for your recommender funcions, etc
- Data: import files/models here e.g.
    - movies: list of movie titles and assigned cluster
    - ratings
    - user_item_matrix
    - item-item matrix
"""
import pandas as pd

movies = pd.read_csv('data/movies.csv')
ratings_reduced = pd.read_csv(r'data/ratings_movies_reduced.csv')

def create_user_vector(user_rating, movies):
    '''
    Convert dict of user_ratings to a user_vector
    '''

    # generate the user vector
    print(user_rating)
    user_vector = None
    return user_vector

def movie_to_id(string_titles):
    '''
    converts movie title to id for use in algorithms
    '''

    movieID = movies.set_index('title').loc[string_titles]['movieId']
    movieID = movieID.tolist()

    return movieID

def id_to_movie(movieID):
    '''
    converts movie Id to title
    '''
    rec_title = movies.set_index('movieId').loc[movieID]['title']

    return rec_title
