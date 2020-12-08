import os
from random import shuffle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

genres = (
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western"
)


class DatasetHandler(object):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def ids2titles(self, ids):
        return [self.id_to_title[movie_id] for movie_id in ids]

    def indices2ids(self, indices):
        return [self.movie_index_to_movie_id[index] for index in indices]

    def id2index(self, movie_id):
        return self.movie_index_to_movie_id.index(movie_id)

    def movie_vector2genres(self, movie_vector):
        return [self.feature_index2genre(i) for i, x in enumerate(movie_vector) if x == 1]

    def feature_index2genre(self, feature_index):
        return genres[feature_index]

    def load_movies(self):
        movies_frame = pd.read_csv(
            os.path.join(self.dataset_path, "movies.csv")
        )
        self.id_to_title = {}
        self.movie_index_to_movie_id = []
        movies_vectors = []
        for _, row in movies_frame.iterrows():
            genres_list = row["genres"].split("|")
            self.id_to_title[int(row["movieId"])] = row["title"]
            self.movie_index_to_movie_id.append(int(row["movieId"]))
            movies_vectors.append(
                np.array([1 if genre in genres_list else 0 for genre in genres])
            )
        return np.array(movies_vectors)

    def load_users_ratings(self):
        ratings_frame = pd.read_csv(
            os.path.join(self.dataset_path, "ratings.csv")
        )
        users_ratings = {}
        for _, row in ratings_frame.iterrows():
            if int(row["userId"]) not in users_ratings:
                users_ratings[int(row["userId"])] = {}
            users_ratings[int(row["userId"])][int(
                row["movieId"])] = row["rating"]
        return users_ratings


class Evaluator(object):
    def __init__(self, recommender):
        self.recommender = recommender

    def compute_map(self, relevant_threshold=3.0, top_n=5):
        k_cross = 5
        total_aps = 0.0
        total = 0
        users_ratings = self.recommender.dataset_handler.load_users_ratings()
        training_data = {
            user: user_ratings for user, user_ratings in users_ratings.items() if user < 0.8*len(users_ratings)
        }
        test_data = {
            user: user_ratings for user, user_ratings in users_ratings.items() if user not in training_data
        }
        self.recommender.train(training_data)
        for user_ratings in test_data.values():
            user_items = user_ratings.items()
            shuffle(user_items)
            parts = [
                user_items[k*(len(user_items)/k_cross):(k+1)*(len(user_items) /
                                                              k_cross) if k < k_cross-1 else len(user_items)]
                for k in range(k_cross)
            ]
            for i in range(k_cross):
                test, training = parts[i], [
                    rat for part in parts[:i]+parts[i+1:] for rat in part]
                relevant = [movie_id for (
                    movie_id, rating) in test if rating >= relevant_threshold]
                user_profile = self.recommender.create_user_profile(
                    dict(training))
                predicted = self.recommender.top(user_profile, top_n=top_n)
                if relevant:
                    total_aps += self._compute_ap(relevant, predicted)
                    total += 1
        return total_aps/total

    def compute_rsme(self):
        k_cross = 5
        rse = 0.0
        total = 0
        users_ratings = self.recommender.dataset_handler.load_users_ratings()
        s = pd.Series(users_ratings)
        training_data, test_data = [i.to_dict()
                                    for i in train_test_split(s, train_size=0.8)]

        # training_data = {user: user_ratings for user, user_ratings in users_ratings.items(
        # ) if user < 0.8*len(users_ratings)}
        # test_data = {user: user_ratings for user,
        #              user_ratings in users_ratings.items() if user not in training_data}
        self.recommender.train(training_data)
        for user_ratings in test_data.values():
            user_items = list(user_ratings.items())
            shuffle(user_items)
            parts = [
                user_items[k*(len(user_items)/k_cross):(k+1)*(len(user_items) /
                                                              k_cross) if k < k_cross-1 else len(user_items)]
                for k in range(k_cross)
            ]
            for i in range(k_cross):
                test, training = parts[i], [
                    rat for part in parts[:i]+parts[i+1:] for rat in part]
                user_profile = self.recommender.create_user_profile(
                    dict(training))
                for (movie_id, rating) in test:
                    predicted = self.recommender.predict_rating(
                        user_profile, movie_id)
                    if predicted > 0:
                        rse += (rating - predicted)**2
                        total += 1
        return rse/total

    def _compute_ap(self, relevant, predicted):
        ap = 0.0
        good_predictions = 0.0
        for i, item in enumerate(predicted):
            if item in relevant:
                good_predictions += 1
                ap += 1.0/(i+1) * good_predictions/(i+1)
        return ap
