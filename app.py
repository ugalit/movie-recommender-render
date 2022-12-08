'''
Movie recommendation website using Flask
choice between 4 algorithms that give recommendations
based on user input , random or popular movies
'''


from flask import Flask, render_template, request
from recommender import recommend_random, recommend_NMF, recommend_neighborhood, recommend_popular
from utils import movies, ratings_reduced, movie_to_id, id_to_movie

app = Flask(__name__)

@app.route('/')
def hello():
    '''Template for main page'''
    return render_template('index.html', name='movie lovers', movies_list=movies.title.to_list())

@app.route('/recommendation')
def recommendation():
    '''ol: ordered list; ul: unordered list; li: list'''

    k = int(request.args.get('k'))
    titles = request.args.getlist('title')
    ratings = request.args.getlist('rating')
    ids = movie_to_id(titles)
    user_rating = dict(zip(titles, ratings))
    query_dict = dict(zip(ids, ratings))

    for keys in user_rating:
        user_rating[keys] = int(user_rating[keys])
    print(user_rating)

    if request.args['method'] == 'Random':
        recs = recommend_random(k=k).to_list()
        return render_template('recommender.html', values = recs)
    elif request.args['method'] == 'Popular':
        recs = recommend_popular(query=query_dict, ratings=ratings_reduced, k=k)
        recs = id_to_movie(recs)
        return render_template('recommender.html', values = recs)
    elif request.args['method']=='Neighborhood':
        recs = recommend_neighborhood(query=query_dict, ratings=ratings_reduced, k=k)
        return render_template('recommender.html', values = recs)
    elif request.args['method']=='NMF':
        recs = recommend_NMF(query=query_dict, ratings=ratings_reduced, k=k)
        recs = id_to_movie(recs)
        return render_template('recommender.html', values = recs)
    else:
        return 'Please select a recommendation algorithm'

# default port that uses http protocol
# debug option: see errors; update page automatically

if __name__ == '__main__':
    app.run(port=5000, debug=True)
