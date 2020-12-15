import os

import pandas as pd


def merge_df_on_movie_id(df1, df2):
    return pd.merge(df1, df2, left_on='movieId', right_on='movieId')


def merge_df_on_index(df1, df2):
    return pd.merge(df1, df2, left_index=True, right_index=True)


def rating_to_df(data):
    rs = pd.DataFrame()
    for k, v in data.items():
        a = pd.DataFrame.from_dict({k: v}, orient='index')
        a.reset_index(level=0, inplace=True)
        a.rename(columns={'index': 'userId'}, inplace=True)
        rs = rs.append(a)
        del a
    return rs.sort_values(by='userId').reset_index(drop=True)


def df_to_tag(df):
    rs = {}
    for index, row in df.iterrows():
        movie_id = int(row['movieId'])
        if not movie_id in rs:
            rs[movie_id] = str(row['tag'])
        elif not str(row['tag']) in rs[movie_id]:
            rs[movie_id] += f";{row['tag']}"
    return rs


def tag_to_df(data):
    rs = pd.DataFrame()
    for k, v in data.items():
        a = pd.DataFrame.from_dict({k: v}, orient='index', columns=['tag'])
        a.reset_index(level=0, inplace=True)
        a.rename(columns={'index': 'movieId'}, inplace=True)
        a1 = a['tag'].str.get_dummies(sep=';')
        a = merge_df_on_index(a, a1)
        rs = rs.append(a)
        del a
    return rs.drop('tag', 1).sort_values(by='movieId').reset_index(drop=True)


def df_to_rating(df):
    rs = {}
    for index, row in df.iterrows():
        user_id = int(row['userId'])
        if not user_id in rs:
            rs[user_id] = {}
        rs[user_id][int(row['movieId'])] = float(row['rating'])
    return rs


def df_to_genome_tag(df):
    rs = {}
    for index, row in df.iterrows():
        movie_id = int(row['movieId'])
        if not movie_id in rs:
            rs[movie_id] = {}
        rs[movie_id][int(row['tagId'])] = float(row['relevance'])
    return rs


def genome_tag_to_df(data):
    rs = pd.DataFrame()
    for k, v in data.items():
        a = pd.DataFrame.from_dict({k: v}, orient='index')
        a.reset_index(level=0, inplace=True)
        a.rename(columns={'index': 'movieId'}, inplace=True)
        rs = rs.append(a)
        del a
    return rs.sort_values(by='movieId').reset_index(drop=True)


def get_dummies_from_genre(df):
    g = df['genres'].str.get_dummies(sep='|')
    return merge_df_on_index(df, g).drop('genres', 1).drop('title', 1)


class Dataset:
    def __init__(self, dataset_path, ext, sep):
        self.dataset_path = dataset_path
        self.ext = ext
        self.sep = sep

    def load_df(self, filename, columns=None):
        if columns:
            return pd.read_csv(
                os.path.join(self.dataset_path, f"{filename}.{self.ext}"),
                sep=self.sep,
                header=None,
                names=columns
            )
        else:
            return pd.read_csv(
                os.path.join(self.dataset_path, f"{filename}.{self.ext}"),
                sep=self.sep
            )

    def save_df(self, df, filename, index=False):
        df.to_csv(
            os.path.join(
                self.dataset_path,
                f"{filename}.{self.ext}"
            ),
            index=index
        )
