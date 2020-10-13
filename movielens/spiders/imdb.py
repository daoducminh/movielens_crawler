# -*- coding: utf-8 -*-

from scrapy import Request, Spider
from scrapy.http.response import Response
import pandas as pd

BASE_URL = 'http://www.imdb.com/title/tt{}/'
STORY_LINE_XPATH = '//div[@id="titleStoryLine"]/div[1]/p/span/text()'
CAST_XPATH = '//div[@id="titleCast"]/table/tr[@class]/td[not(@class)]/a/text()'

movies = ['0114709', '0112302']


class ImdbSpider(Spider):
    name = 'imdb'

    custom_settings = {
        'ITEM_PIPELINES': {
            'movielens.pipelines.MovielensPipeline': 100,
        },
        'LOG_ENABLED': False,
        'DEFAULT_REQUEST_HEADERS': {
            'accept': '*/*'
        }
    }

    def start_requests(self):
        df = pd.read_csv('links.csv')
        for index, row in df.iterrows():
            yield Request(
                url=BASE_URL.format(row['imbdId']),
                cb_kwargs=dict(imdb_id=row['imbdId'])
            )

    def parse(self, response: Response, imdb_id):
        story_line_element = response.xpath(STORY_LINE_XPATH)
        cast_elements = response.xpath(CAST_XPATH)
        yield {
            'imdb_id': imdb_id,
            'story_line': story_line_element.get().strip(),
            'cast': '|'.join([a.get().strip() for a in cast_elements])
        }