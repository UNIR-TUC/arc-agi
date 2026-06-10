# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class ArcAgiDataScraperItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    pass


class LeaderboardItem(scrapy.Item):
    model = scrapy.Field()
    author = scrapy.Field()
    date = scrapy.Field()
    solution_type = scrapy.Field()
    arc_agi_1 = scrapy.Field()   # float (%) or None
    arc_agi_2 = scrapy.Field()   # float (%) or None
    arc_agi_3 = scrapy.Field()   # float (%) or None
    cost_per_task = scrapy.Field()  # float ($) or None
    total_cost = scrapy.Field()     # str or None
