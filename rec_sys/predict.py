from rec_sys.recommenders import UserBasedCFRecommender, UserBasedCFWithClusteringRecommender, ContentBasedCFRecommender
from rec_sys.helper import Evaluator, DatasetHandler

DATESET_1M = 'ml-1m'
DATESET_SMALL = 'ml-latest-small'


def user_based_cf(dataset_handler):
    recommender = UserBasedCFRecommender(dataset_handler, 20)
    users_ratings = dataset_handler.load_users_ratings()
    user_ratings = users_ratings[1]
    del users_ratings[1]
    recommender.train(users_ratings)
    user_profile = recommender.create_user_profile(user_ratings)
    recommender.present_user(user_profile, user_ratings)
    top = recommender.top(user_profile, top_n=5)
    recommender.present_recommendations(top)


def user_based_cf_wth_clustering(dataset_handler):
    recommender = UserBasedCFWithClusteringRecommender(
        dataset_handler, 5, 10)
    users_ratings = dataset_handler.load_users_ratings()
    user_ratings = users_ratings[1]
    del users_ratings[1]
    recommender.train(users_ratings)
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
    user_based_cf_wth_clustering(dataset)
    content_based_cf(dataset)
