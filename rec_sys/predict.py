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
        # return recommender.mae(user_ratings, k_neighbor)


def content_based_on_tags(dataset, movie_id):
    recommender = ContentBasedRecommender(dataset)
    # return recommender.top_cos_genome_tag(movie_id, top_n=10)
    return recommender.top_cos_tag(movie_id, top_n=10)


if __name__ == '__main__':
    # data = Dataset(DATESET_SMALL, 'csv', ',')
    data = Dataset(DATASET_1M, 'csv', ',')

    # Content based
    # print(content_based_on_tags(data, movie_id=246))

    # User based
    # rate = {}
    # ids = (1, 3, 6, 47, 50, 70, 101, 110, 151, 157, 163, 216, 223, 231, 235)
    # sc = (4.0, 4.0, 4.0, 5.0, 5.0, 3.0, 5.0,
    #       5.0, 4.0, 4.0, 4.5, 4.5, 3.5, 5.0, 4.5)
    # for i in range(len(ids)):
    #     rate[ids[i]] = sc[i]
    # print(user_based_cf(data, rate, top_n=10, k_neighbor=50))

    movies = list(data.load_df('movies').loc[:, 'movieId'])
    rate = {}
    for i in range(50):
        rate[movies[i]] = SCORES[random.randrange(0, len(SCORES), 1)]
    print(user_based_cf(data, rate, top_n=10, k_neighbor=50))
