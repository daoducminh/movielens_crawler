from rec_sys.recommenders import *
from rec_sys.helper import Evaluator, DatasetHandler

DATESET_1M = 'ml-1m'


def recommend_cf(dataset_handler):
    recommender = CollaborativeFilteringRecommender(dataset_handler, 20)
    users_ratings = dataset_handler.load_users_ratings()
    user_ratings = users_ratings[1]
    del users_ratings[1]
    recommender.train(users_ratings)
    user_profile = recommender.create_user_profile(user_ratings)
    recommender.present_user(user_profile, user_ratings)
    top = recommender.top(user_profile, top_n=5)
    recommender.present_recommendations(top)
    evaluator = Evaluator(recommender)
    evaluator.compute_rsme()
    evaluator.compute_map()


if __name__ == '__main__':
    dataset = DatasetHandler(DATESET_1M)
    recommend_cf(dataset)
