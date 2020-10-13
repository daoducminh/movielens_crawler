# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


class MovielensPipeline:
    def __init__(self):
        self.file = None

    def open_spider(self, spider):
        self.file = open('cast.csv', 'w')
        self.file.writelines('imdbId,storyLine,cast\n')

    def close_spider(self, spider):
        self.file.close()

    def process_item(self, item, spider):
        self.file.writelines(
            f'{item["imdb_id"]},"{item["story_line"]}",{item["cast"]}\n'
        )
        return item
