import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


class UserBasedCFRecommender:
    def __init__(self, dataset_handler, train_set, neighbours_to_predict=5):
        self.dataset_handler = dataset_handler
        self.movies_vectors = self.dataset_handler.load_movies()
        self.movies_ids = set(self.dataset_handler.id_to_title.keys())
        self.neighbours_to_predict = neighbours_to_predict
        self.users_ratings = train_set
        self.users_profiles, self.user_id_to_profile_index = self._create_users_profiles(
            train_set)
        self.movies_watchers = self._get_movies_watchers()
        self.nbrs = NearestNeighbors(metric='cosine', algorithm='brute')

    def top(self, user_profile, top_n):
        unrated_movies = np.array([
            (movie_id, self.predict_rating(user_profile, movie_id))
            for movie_id in list(self.movies_ids - user_profile[1])
        ])
        return unrated_movies[np.argpartition(-unrated_movies[:, 1], top_n)[:top_n], 0]

    def predict_rating(self, user_profile, movie_id):
        profiles_with_ids = np.array([
            np.hstack(
                [[watcher], self.users_profiles[self.user_id_to_profile_index[watcher]][0]])
            for watcher in self.movies_watchers[movie_id]
        ])
        nearest_neighbours = self._cosine_knn(
            user_profile, profiles_with_ids, self.neighbours_to_predict)
        if not nearest_neighbours:
            return 0.0
        return np.average([self.users_ratings[neighbour][movie_id] for neighbour in nearest_neighbours])

    def create_user_profile(self, user_ratings):
        mid_rating = 2.75
        profile = np.average(
            np.array([
                self.movies_vectors[self.dataset_handler.id2index(
                    movie)] * np.sign(rating - mid_rating)
                for (movie, rating) in user_ratings.items()
            ]),
            weights=(np.array(list(user_ratings.values())) - mid_rating) ** 2,
            axis=0
        )
        watched_movies = set(user_ratings.keys())
        return (profile, watched_movies)

    def present_user(self, user_profile, user_ratings):
        print("User favourite genre:", self.dataset_handler.feature_index2genre(
            np.argmax(user_profile[0])))
        print("User ratings:")
        for (movie_id, rating) in user_ratings.items():
            movie_vector = self.movies_vectors[self.dataset_handler.id2index(
                movie_id)]
            print("{} {}: {}".format(
                self.dataset_handler.id_to_title[movie_id],
                self.dataset_handler.movie_vector2genres(movie_vector),
                rating
            ))

    def present_recommendations(self, recommendations):
        print("Recommended movies:")
        for movie_id in recommendations:
            movie_vector = self.movies_vectors[self.dataset_handler.id2index(
                movie_id)]
            print("{} {}".format(
                self.dataset_handler.id_to_title[movie_id],
                self.dataset_handler.movie_vector2genres(movie_vector)
            ))

    def _get_movies_watchers(self):
        movies_watchers = {}
        for (user, user_ratings) in self.users_ratings.items():
            for movie_id in user_ratings.keys():
                movies_watchers[movie_id].append(user)
        return movies_watchers

    def _create_users_profiles(self, users_ratings):
        users_profiles = []
        user_id_to_profile_index = {}
        for i, (user, user_ratings) in enumerate(users_ratings.items()):
            users_profiles.append(self.create_user_profile(user_ratings))
            user_id_to_profile_index[user] = i
        return users_profiles, user_id_to_profile_index

    def _cosine_knn(self, user_profile, profiles_with_ids, k, treshold=20):
        if profiles_with_ids.shape[0] < treshold:
            return []
        self.nbrs.fit(profiles_with_ids[:, 1:])
        return [
            profiles_with_ids[i, 0]
            for i in self.nbrs.kneighbors(
                np.array([user_profile[0]]),
                n_neighbors=min(k, len(profiles_with_ids)),
                return_distance=False
            )[0]
        ]


class ContentBasedCFRecommender:
    def __init__(self, dataset_handler):
        self.dataset_handler = dataset_handler
        self.movies_vectors = self.dataset_handler.load_movies()

    def top(self, user_profile, top_n):
        return self._cosine_knn_all_movies(user_profile[0], top_n)

    def create_user_profile(self, user_ratings):
        return (
            np.average(
                np.array([
                    self.movies_vectors[self.dataset_handler.id2index(movie)]
                    for (movie, rating) in user_ratings.items()
                ]),
                weights=np.array(list(user_ratings.values())),
                axis=0
            ),
            user_ratings
        )

    def present_user_profile(self, user_profile):
        print("User favourite genre:", self.dataset_handler.feature_index2genre(
            np.argmax(user_profile[0])))
        print("User ratings:")
        for (movie_id, rating) in user_profile[1].items():
            movie_vector = self.movies_vectors[self.dataset_handler.id2index(
                movie_id)]
            print("{} {}: {}".format(
                self.dataset_handler.id_to_title[movie_id],
                self.dataset_handler.movie_vector2genres(movie_vector),
                rating
            ))

    def present_recommendations(self, recommendations):
        print("Recommended movies:")
        for movie_id in recommendations:
            movie_vector = self.movies_vectors[self.dataset_handler.id2index(
                movie_id)]
            print("{} {}".format(
                self.dataset_handler.id_to_title[movie_id],
                self.dataset_handler.movie_vector2genres(movie_vector)
            ))

    def _cosine_knn_all_movies(self, user_profile, k):
        nbrs = NearestNeighbors(metric='cosine', algorithm='brute')
        nbrs.fit(self.movies_vectors)
        return self.dataset_handler.indices2ids(nbrs.kneighbors(np.array([user_profile]), k, return_distance=False)[0])
