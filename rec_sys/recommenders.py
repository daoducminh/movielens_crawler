import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from rec_sys.helper import Dataset, merge_df_on_index, get_dummies_from_genre


class Recommender:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.movie_df = self.dataset.load_df('movies').set_index('movieId')


class UserBasedCFRecommender(Recommender):
    def __init__(self, dataset: Dataset):
        super().__init__(dataset)
        self.rating_df = self.dataset.load_df('ratings1').set_index('userId')

    def predict_movie_score(self, movie_id, top_df):
        score = top_df.loc[:, str(movie_id)].mean()
        return score

    def top_movies(self, user_ratings, top_n=10, k_neighbor=50):
        user_id = self.rating_df.index.max() + 1
        id_series = self.rating_df.index
        movie_list = list(id_series)
        ur_d = {
            user_id: user_ratings
        }
        ur_df = pd.DataFrame.from_dict(ur_d, orient='index')
        rating_df = self.rating_df.append(ur_df).replace(0, np.NaN)

        avg = rating_df.mean(axis=1)
        n_rating_df = rating_df.subtract(avg, axis=0)
        n1_rating_df = n_rating_df.replace(np.NaN, 0)
        x_array = n1_rating_df.loc[user_id, :].to_numpy()
        y_array = n1_rating_df.loc[:user_id-1, :].to_numpy()

        t = cosine_similarity(x_array.reshape(1, -1), y_array)
        cosine_df = pd.DataFrame(
            {'userId': id_series, 'cos': t[0]}
        ).set_index('userId')

        user_cos_df = merge_df_on_index(self.rating_df, cosine_df)
        user_cos_df = user_cos_df.sort_values('cos', ascending=False)

        top_df = user_cos_df.head(k_neighbor)

        rs = {}
        for i in list(rating_df):
            if i in movie_list:
                if not pd.isna(rating_df.loc[user_id, i]):
                    score = self.predict_movie_score(i, top_df)
                if score and not np.isnan(score):
                    rs[i] = score
        if rs:
            a = pd.DataFrame.from_dict(
                rs, orient='index'
            ).rename(columns={0: 'cos'})

            b = merge_df_on_index(self.movie_df, a).sort_values(
                'cos', ascending=False)
            return b.head(top_n).to_dict(orient='index')
        else:
            return []


class ContentBasedRecommender(Recommender):
    def __init__(self, dataset: Dataset):
        super().__init__(dataset)
        self.genre_df = get_dummies_from_genre(self.movie_df)
        self.tag_df = self.dataset.load_df('tags1').set_index('movieId')

    def top_cos_tag(self, movie_id, top_n=10, tag_weight=1, genre_weight=1):
        if movie_id in self.tag_df.index:
            tg_df = merge_df_on_index(self.tag_df, self.genre_df)
            df = merge_df_on_index(tg_df, self.movie_df)[['title']]
            id_series = tg_df.index

            tag_vec_df = self.tag_df.replace(np.NaN, 0).mul(tag_weight)
            genre_vec_df = self.genre_df.replace(np.NaN, 0).mul(genre_weight)
            vector_df = merge_df_on_index(tag_vec_df, genre_vec_df)

            x_array = vector_df.loc[movie_id].to_numpy()
            y_array = vector_df.to_numpy()

            t = cosine_similarity(x_array.reshape(1, -1), y_array)
            cosine_df = pd.DataFrame(
                {'movieId': id_series, 'cos': t[0]}
            ).set_index('movieId')

            movie_cos_df = merge_df_on_index(df, cosine_df)
            movie_cos_df = movie_cos_df.drop(index=movie_id).sort_values(
                'cos', ascending=False)
            return movie_cos_df.head(top_n).to_dict(orient='index')
        else:
            return []

    def top_cos_genome_tag(self, movie_id, top_n, tag_weight=1, genre_weight=1):
        pass
