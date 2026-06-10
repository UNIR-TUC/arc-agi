"""
Scrapy spider that scrapes the ARC Prize leaderboard table from
https://arcprize.org/leaderboard

The page is a React app, so scrapy-playwright is used to wait for the
table to be fully rendered before parsing.

Required packages:
    pip install scrapy-playwright
    playwright install chromium
"""

import re
import scrapy
from scrapy_playwright.page import PageMethod

from arc_agi_data_scraper.items import LeaderboardItem


class LeaderboardSpider(scrapy.Spider):
    name = "leaderboard"
    allowed_domains = ["arcprize.org"]
    start_urls = ["https://arcprize.org/leaderboard"]

    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(
                url,
                meta={
                    "playwright": True,
                    "playwright_page_methods": [
                        # Wait until at least one data row is visible
                        PageMethod("wait_for_selector", "table tbody tr td"),
                    ],
                },
            )

    def parse(self, response):
        """
        Parse the leaderboard breakdown table.

        Column order (0-indexed):
            0  model
            1  author
            2  date
            3  solution_type
            4  arc_agi_1   (%)
            5  arc_agi_2   (%)
            6  arc_agi_3   (%)
            7  cost_per_task  ($)
            8  total_cost
            9  report icon  (ignored)
        """
        rows = response.css("table tbody tr")

        if not rows:
            self.logger.warning(
                "No table rows found. The page may not have rendered "
                "correctly. Check playwright configuration."
            )
            return

        self.logger.info("Found %d table rows", len(rows))

        for row in rows:
            cells = row.css("td")
            if len(cells) < 9:
                # Skip header-like rows or incomplete rows
                continue

            item = LeaderboardItem()
            item["model"] = self._text(cells[0])
            item["author"] = self._text(cells[1])
            item["date"] = self._text(cells[2])
            item["solution_type"] = self._text(cells[3])
            item["arc_agi_1"] = self._text(cells[4])
            item["arc_agi_2"] = self._text(cells[5])
            item["arc_agi_3"] = self._text(cells[6])
            item["cost_per_task"] = self._text(cells[7])
            item["total_cost"] = self._text(cells[8])

            yield item

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _text(selector) -> str:
        """Join all text nodes so footnote superscripts are not lost.
        The pipeline will normalise the result."""
        return " ".join(
            t.strip() for t in selector.css("::text").getall() if t.strip()
        )
