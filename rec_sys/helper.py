import os
from random import shuffle
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def convert_ratings(filename):
    with open(filename, 'r') as csv_file:
        rs = {}
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for row in csv_reader:
            user_id = int(row['userId'])
            if not user_id in rs:
                rs[user_id] = {}
            rs[user_id][int(row['movieId'])] = float(row['rating'])
        return rs


def rating_to_df(data):
    rs = pd.DataFrame()
    for k, v in data.items():
        a = pd.DataFrame.from_dict({k: v}, orient='index')
        a.reset_index(level=0, inplace=True)
        a.rename(columns={'index': 'userId'}, inplace=True)
        rs = rs.append(a)
        del a
    return rs.sort_values(by='userId').reset_index(drop=True)


def convert_tags(filename):
    with open(filename, 'r') as csv_file:
        rs = {}
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for row in csv_reader:
            movie_id = int(row['movieId'])
            if not movie_id in rs:
                rs[movie_id] = row['tag']
            elif not row['tag'] in rs[movie_id]:
                rs[movie_id] += f";{row['tag']}"
        return rs


def tag_to_df(data):
    rs = pd.DataFrame()
    for k, v in data.items():
        a = pd.DataFrame.from_dict({k: v}, orient='index', columns=['tag'])
        a.reset_index(level=0, inplace=True)
        a.rename(columns={'index': 'movieId'}, inplace=True)
        a1 = a['tag'].str.get_dummies(sep=';')
        a = pd.merge(a, a1, left_index=True, right_index=True)
        del a
    return rs.sort_values(by='movieId').reset_index(drop=True)


def merge_df_on_movie_id(df1, df2):
    return pd.merge(df1, df2, left_on='movieId', right_on='movieId')


def merge_df_on_index(df1, df2):
    return pd.merge(df1, df2, left_index=True, right_index=True)


class Dataset:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def load_df(self, filename):
        return pd.read_csv(
            os.path.join(self.dataset_path, f"{filename}.csv")
        )

    def save_df(self, df, filename):
        df.to_csv(
            os.path.join(
                self.dataset_path,
                f"{filename}.csv"
            ),
            index=False
        )

    def load_users_ratings(self):
        ratings_df = self.load_df('ratings')
        users_ratings = {}
        for _, row in ratings_df.iterrows():
            if int(row["userId"]) not in users_ratings:
                users_ratings[int(row["userId"])] = {}
            users_ratings[int(row["userId"])][int(
                row["movieId"])] = row["rating"]
        return users_ratings


# class DatasetHandler(object):
#     def __init__(self, dataset_path):
#         self.dataset_path = dataset_path

#     def indices2ids(self, indices):
#         return [self.movie_index_to_movie_id[index] for index in indices]

#     def id2index(self, movie_id):
#         return self.movie_index_to_movie_id.index(movie_id)

#     def movie_vector2genres(self, movie_vector):
#         return [self.feature_index2genre(i) for i, x in enumerate(movie_vector) if x == 1]

#     def feature_index2genre(self, feature_index):
#         return genres[feature_index]

#     def load_movies(self):
#         movies_frame = pd.read_csv(
#             os.path.join(self.dataset_path, "movies.csv")
#         )
#         self.id_to_title = {}
#         self.movie_index_to_movie_id = []
#         movies_vectors = []
#         for _, row in movies_frame.iterrows():
#             genres_list = row["genres"].split("|")
#             self.id_to_title[int(row["movieId"])] = row["title"]
#             self.movie_index_to_movie_id.append(int(row["movieId"]))
#             movies_vectors.append(
#                 np.array([1 if genre in genres_list else 0 for genre in genres])
#             )
#         return np.array(movies_vectors)

#     def load_users_ratings(self):
#         ratings_frame = pd.read_csv(
#             os.path.join(self.dataset_path, "ratings.csv")
#         )
#         users_ratings = {}
#         for _, row in ratings_frame.iterrows():
#             if int(row["userId"]) not in users_ratings:
#                 users_ratings[int(row["userId"])] = {}
#             users_ratings[int(row["userId"])][int(
#                 row["movieId"])] = row["rating"]
#         return users_ratings
