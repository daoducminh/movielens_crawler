from rec_sys.recommenders import UserBasedCFRecommender, ContentBasedRecommender
from rec_sys.helper import Dataset

DATESET_1M = 'ml-1m'
DATESET_SMALL = 'data/ml-latest-small'


# def user_based_cf(dataset, user_id, user_ratings=None, top_n=5, k_neighbor=20):
#     users_ratings = dataset.load_users_ratings()
#     if user_id in users_ratings:
#         user_ratings = users_ratings[user_id]
#         del users_ratings[user_id]

#     recommender = UserBasedCFRecommender(
#         dataset, users_ratings, k_neighbor)

#     user_profile = recommender.create_user_profile(user_ratings)
#     recommender.present_user(user_profile, user_ratings)

#     top = recommender.top(user_profile, top_n=top_n)
#     recommender.present_recommendations(top)


def content_based_cf(dataset, movie_id):
    recommender = ContentBasedRecommender(dataset)
    top = recommender.top_cos_tag(movie_id, top_n=10)
    print(top)


if __name__ == '__main__':
    data = Dataset(DATESET_SMALL)
    # user_based_cf(data, 1)
    content_based_cf(data, 3)
