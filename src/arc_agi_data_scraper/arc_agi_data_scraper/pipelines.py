# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


import re
from itemadapter import ItemAdapter


# Sentinel values that mean "no data" in the leaderboard table
_EMPTY_VALUES = {"n/a", "—", "-", ""}

# Fields that contain a percentage value  (e.g. "92.0%", "1.5%")
_PERCENT_FIELDS = {"arc_agi_1", "arc_agi_2", "arc_agi_3"}

# Fields that contain a dollar amount  (e.g. "$2.74", "$10.0K")
_DOLLAR_FIELDS = {"cost_per_task"}


class ArcAgiDataScraperPipeline:
    def process_item(self, item, spider):
        return item


class LeaderboardCleaningPipeline:
    """
    Normalise raw cell text extracted by LeaderboardSpider:

    * Percentage fields  → float (0–100) or None
    * cost_per_task      → float (USD) or None   ($10.0K → 10000.0)
    * total_cost         → cleaned string or None ($8.9K / $10.0K / N/A)
    * All other fields   → stripped string, footnote markers removed
    """

    # Match a leading footnote superscript such as "¹", "²", etc.
    _FOOTNOTE_RE = re.compile(r"\s*[¹²³⁴⁵⁶⁷⁸⁹]+\s*$")

    def process_item(self, item, spider):
        adapter = ItemAdapter(item)

        for field in _PERCENT_FIELDS:
            adapter[field] = self._parse_percent(adapter.get(field))

        adapter["cost_per_task"] = self._parse_dollar(adapter.get("cost_per_task"))
        adapter["total_cost"] = self._clean_total_cost(adapter.get("total_cost"))

        for field in ("model", "author", "date", "solution_type"):
            raw = adapter.get(field) or ""
            adapter[field] = self._FOOTNOTE_RE.sub("", raw).strip() or None

        return item

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_percent(self, value) -> float | None:
        """'92.0%' → 92.0, 'N/A' → None, '—' → None."""
        if not value or value.strip().lower() in _EMPTY_VALUES:
            return None
        cleaned = value.replace("%", "").strip()
        # Remove any trailing footnote markers (digits, superscript digits)
        cleaned = re.sub(r"[¹²³⁴⁵⁶⁷⁸⁹\d]+$", "", cleaned).strip()
        try:
            return float(cleaned)
        except ValueError:
            return None

    def _parse_dollar(self, value) -> float | None:
        """'$2.74' → 2.74, '$10.0K' → 10000.0, 'N/A' → None."""
        if not value or value.strip().lower() in _EMPTY_VALUES:
            return None
        cleaned = value.replace("$", "").strip()
        multiplier = 1.0
        if cleaned.upper().endswith("K"):
            multiplier = 1_000.0
            cleaned = cleaned[:-1]
        elif cleaned.upper().endswith("M"):
            multiplier = 1_000_000.0
            cleaned = cleaned[:-1]
        try:
            return float(cleaned) * multiplier
        except ValueError:
            return None

    def _clean_total_cost(self, value) -> str | None:
        """Keep the raw string but normalise empties to None."""
        if not value or value.strip().lower() in _EMPTY_VALUES:
            return None
        return value.strip()
