from flask import Flask, render_template, request
from recommender import recommend_random, recommend_with_NMF, recommend_neighborhood, recommend_popular
from utils import movies, ratings_reduced, movie_to_id, id_to_movie

'''
Flask: super lightweight
alternative: Jinja: double curly brackets
streamlit
'''

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('index.html', name='Silke', movies_list=movies.title.to_list())

@app.route('/recommendation')
def random_recommendation():
    '''ol: ordered list; ul: unordered list; li: list'''

    titles = request.args.getlist('title')
    ratings = request.args.getlist('rating')
    ids = movie_to_id(titles)
    user_rating = dict(zip(titles, ratings))
    query_dict = dict(zip(ids, ratings))

    for keys in user_rating:
        user_rating[keys] = int(user_rating[keys])
    print(user_rating)

    if request.args['method'] == 'Random':
        recs = recommend_random(k=5).to_list()
        return render_template('recommender.html', values = recs)
    elif request.args['method'] == 'Popular':
        recs = recommend_popular(query=query_dict, ratings=ratings_reduced)
        recs = id_to_movie(recs)
        return render_template('recommender.html', values = recs)
    elif request.args['method']=='NMF':
        recs = recommend_with_NMF(user_rating, k=5)
        return 'Function not yet defined'
    elif request.args['method']=='Neighborhood':
        recs = recommend_neighborhood(user_rating, model='model', k=5)
        return 'Function not yet defined'
    else:
        return 'Function not defined'

if __name__ == '__main__':
    '''
    default port that uses http protocol
    debug option: see errors; update page automatically
    '''
    app.run(port=5000, debug=True) 
