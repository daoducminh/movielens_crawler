from rec_sys.recommenders import UserBasedCFRecommender, ContentBasedRecommender
from rec_sys.helper import Dataset, df_to_tag

DATESET_1M = 'ml-1m'
DATESET_SMALL = 'data/ml-latest-small'


def user_based_cf(dataset, user_ratings, movie_id, top_n=10, k_neighbor=20):
    if len(user_ratings) < 10:
        return content_based_on_tags(dataset, movie_id)
    else:
        recommender = UserBasedCFRecommender(dataset)
        return recommender.top_movies(user_ratings, movie_id)


def content_based_on_tags(dataset, movie_id):
    recommender = ContentBasedRecommender(dataset)
    return recommender.top_cos_tag(movie_id, top_n=10)


if __name__ == '__main__':
    data = Dataset(DATESET_SMALL, 'csv', ',')
    # a = df_to_tag(data.load_df('tags'))
    a = (content_based_on_tags(data, 2))
    print(a)
    rate = {
        2: 3.5,
        3: 4.5,
        4: 5.0,
        5: 1.0,
        6: 0.5,
        7: 1.5,
        8: 2.0,
        9: 2.5,
        10: 3.0,
        11: 4.0,
        12: 4.0
    }
    print(user_based_cf(data, rate, 1))
