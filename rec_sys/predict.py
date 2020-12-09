from rec_sys.recommenders import UserBasedCFRecommender, ContentBasedCFRecommender
from rec_sys.helper import DatasetHandler

DATESET_1M = 'ml-1m'
DATESET_SMALL = 'ml-latest-small'


def user_based_cf(dataset_handler):
    users_ratings = dataset_handler.load_users_ratings()
    user_ratings = users_ratings[1]
    del users_ratings[1]

    recommender = UserBasedCFRecommender(dataset_handler, users_ratings, 20)

    user_profile = recommender.create_user_profile(user_ratings)
    recommender.present_user(user_profile, user_ratings)

    top = recommender.top(user_profile, top_n=5)
    recommender.present_recommendations(top)


def content_based_cf(dataset_handler):
    recommender = ContentBasedCFRecommender(dataset_handler)
    user_ratings = dataset_handler.load_users_ratings()

    user_profile = recommender.create_user_profile(user_ratings[1])
    recommender.present_user_profile(user_profile)

    top = recommender.top(user_profile, top_n=5)
    recommender.present_recommendations(top)


if __name__ == '__main__':
    dataset = DatasetHandler(DATESET_SMALL)
    user_based_cf(dataset)
    # content_based_cf(dataset)
