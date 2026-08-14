<<<<<<< HEAD
# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter


class BooksCrawlerPipeline:
    def process_item(self, item):
        return item
=======
# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter


class BooksCrawlerPipeline:
    def process_item(self, item):
        return item
>>>>>>> d6f88a4d189e41fe2a496010b2c32c80695dc817
