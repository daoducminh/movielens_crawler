import random
from rec_sys.helper import Dataset
from rec_sys.recommenders import UserBasedCFRecommender, ContentBasedRecommender

DATASET_1M = 'data/ml-1m'
DATESET_SMALL = 'data/ml-latest-small'
SCORES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]


def user_based_cf(dataset, user_ratings, top_n=10, k_neighbor=20):
    if len(user_ratings) < 10:
        return []
    else:
        recommender = UserBasedCFRecommender(dataset)
        return recommender.top_movies(user_ratings, top_n, k_neighbor)


def user_based_cf_mae(dataset, user_ratings, top_n=10, k_neighbor=20):
    recommender = UserBasedCFRecommender(dataset)
    return recommender.mae(user_ratings, k_neighbor)


def content_based_on_tags(dataset, movie_id):
    recommender = ContentBasedRecommender(dataset)
    return recommender.top_cos_tag(movie_id, top_n=10)


def content_based_on_tag_genome(dataset, movie_id):

    recommender = ContentBasedRecommender(dataset)
    return recommender.top_cos_genome_tag(movie_id, top_n=10)


if __name__ == '__main__':
    data = Dataset(DATESET_SMALL, 'csv', ',')

    # Content based
    print('Content based on tags')
    print(content_based_on_tags(data, movie_id=246))
    print('\nContent based on tag genome')
    print(content_based_on_tag_genome(data, movie_id=246))

    # User based
    movies = list(data.load_df('movies').loc[:, 'movieId'])
    rate = {}
    for i in range(50):
        rate[movies[i]] = SCORES[random.randrange(0, len(SCORES), 1)]
    print('\nUser based CF')
    print(user_based_cf(data, rate, top_n=10, k_neighbor=50))
    print('\nMAE evaluation on User based CF')
    print(user_based_cf_mae(data, rate, top_n=10, k_neighbor=50))
