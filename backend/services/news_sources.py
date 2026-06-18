"""
News Sources — static registry of 56 public RSS/Atom feeds across 8 categories.
"""

from collections import namedtuple
from typing import Optional

Source = namedtuple("Source", "id name category feed_url country")

SOURCES = (
    # ── International (10) ────────────────────────────────────
    Source("bbc_world", "BBC World News", "international", "https://feeds.bbci.co.uk/news/world/rss.xml", "GB"),
    Source("reuters_world", "Reuters World", "international", "https://feeds.reuters.com/Reuters/worldNews", "US"),
    Source("ap_world", "AP News World", "international", "https://rsshub.app/apnews/topics/apf-topnews", "US"),
    Source("aljazeera", "Al Jazeera", "international", "https://aljazeera.com/xml/rss/all.xml", "QA"),
    Source("dw_world", "DW News", "international", "https://rss.dw.com/rdf/rss-en-all", "DE"),
    Source("france24_en", "France 24 English", "international", "https://france24.com/en/rss", "FR"),
    Source("euronews", "Euronews", "international", "https://euronews.com/rss", "FR"),
    Source("nhk_world", "NHK World", "international", "https://www3.nhk.or.jp/rss/news/cat0.xml", "JP"),
    Source("abc_intl", "ABC News International", "international", "https://abcnews.go.com/abcnews/internationalheadlines", "US"),
    Source("cbc_world", "CBC World", "international", "https://rss.cbc.ca/lineup/world.xml", "CA"),

    # ── US (10) ───────────────────────────────────────────────
    Source("cnn", "CNN", "us", "https://rss.cnn.com/rss/cnn_topstories.rss", "US"),
    Source("nytimes", "New York Times", "us", "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml", "US"),
    Source("wapo", "Washington Post", "us", "https://feeds.washingtonpost.com/rss/national", "US"),
    Source("npr", "NPR News", "us", "https://feeds.npr.org/1001/rss.xml", "US"),
    Source("fox_news", "Fox News", "us", "https://feeds.foxnews.com/foxnews/latest", "US"),
    Source("usa_today", "USA Today", "us", "https://rssfeeds.usatoday.com/usatoday-NewsTopStories", "US"),
    Source("chicago_tribune", "Chicago Tribune", "us", "https://chicagotribune.com/arcio/rss/category/nation-world/", "US"),
    Source("la_times", "Los Angeles Times", "us", "https://latimes.com/world-nation/rss2.0.xml", "US"),
    Source("politico", "Politico", "us", "https://rss.politico.com/politics-news.xml", "US"),
    Source("hill", "The Hill", "us", "https://thehill.com/feed/", "US"),

    # ── UK (6) ────────────────────────────────────────────────
    Source("bbc_uk", "BBC UK News", "uk", "https://feeds.bbci.co.uk/news/uk/rss.xml", "GB"),
    Source("guardian", "The Guardian", "uk", "https://feeds.guardian.co.uk/theguardian/rss", "GB"),
    Source("telegraph", "The Telegraph", "uk", "https://telegraph.co.uk/rss.xml", "GB"),
    Source("independent", "The Independent", "uk", "https://independent.co.uk/rss", "GB"),
    Source("sky_news", "Sky News", "uk", "https://feeds.skynews.com/feeds/rss/home.xml", "GB"),
    Source("times_uk", "The Times UK", "uk", "https://thetimes.co.uk/rss", "GB"),

    # ── Tech (8) ──────────────────────────────────────────────
    Source("techcrunch", "TechCrunch", "tech", "https://feeds.feedburner.com/TechCrunch", "US"),
    Source("the_verge", "The Verge", "tech", "https://theverge.com/rss/index.xml", "US"),
    Source("wired", "Wired", "tech", "https://wired.com/feed/rss", "US"),
    Source("ars_technica", "Ars Technica", "tech", "https://feeds.arstechnica.com/arstechnica/index", "US"),
    Source("hacker_news", "Hacker News", "tech", "https://hnrss.org/frontpage", "US"),
    Source("engadget", "Engadget", "tech", "https://engadget.com/rss.xml", "US"),
    Source("zdnet", "ZDNet", "tech", "https://zdnet.com/news/rss.xml", "US"),
    Source("mit_tech_review", "MIT Technology Review", "tech", "https://technologyreview.com/feed", "US"),

    # ── Business (6) ──────────────────────────────────────────
    Source("reuters_business", "Reuters Business", "business", "https://feeds.reuters.com/reuters/businessNews", "US"),
    Source("bloomberg", "Bloomberg", "business", "https://bloomberg.com/feed/podcast/etf-report.xml", "US"),
    Source("ft", "Financial Times", "business", "https://ft.com/rss/home", "GB"),
    Source("wsj", "Wall Street Journal", "business", "https://feeds.a.dj.com/rss/RSSWorldNews.xml", "US"),
    Source("fortune", "Fortune", "business", "https://fortune.com/feed", "US"),
    Source("marketwatch", "MarketWatch", "business", "https://feeds.marketwatch.com/marketwatch/topstories", "US"),

    # ── Science (5) ───────────────────────────────────────────
    Source("nasa", "NASA", "science", "https://nasa.gov/rss/dyn/breaking_news.rss", "US"),
    Source("new_scientist", "New Scientist", "science", "https://newscientist.com/section/news/feed", "GB"),
    Source("science_daily", "ScienceDaily", "science", "https://sciencedaily.com/rss/all.rss", "US"),
    Source("nature", "Nature", "science", "https://nature.com/nature.rss", "GB"),
    Source("scientific_american", "Scientific American", "science", "https://rss.sciam.com/ScientificAmerican-Global", "US"),

    # ── Sports (6) ────────────────────────────────────────────
    Source("bbc_sport", "BBC Sport", "sports", "https://feeds.bbci.co.uk/sport/rss.xml", "GB"),
    Source("espn", "ESPN", "sports", "https://espn.com/espn/rss/news", "US"),
    Source("sky_sports", "Sky Sports", "sports", "https://skysports.com/rss/12040", "GB"),
    Source("sports_illustrated", "Sports Illustrated", "sports", "https://si.com/rss/si_topstories.rss", "US"),
    Source("bleacher_report", "Bleacher Report", "sports", "https://bleacherreport.com/articles/feed", "US"),
    Source("the_athletic", "The Athletic", "sports", "https://theathletic.com/feeds/rss/news/", "US"),

    # ── Entertainment (5) ─────────────────────────────────────
    Source("variety", "Variety", "entertainment", "https://variety.com/feed", "US"),
    Source("entertainment_weekly", "Entertainment Weekly", "entertainment", "https://ew.com/feed", "US"),
    Source("hollywood_reporter", "The Hollywood Reporter", "entertainment", "https://hollywoodreporter.com/feed", "US"),
    Source("rolling_stone", "Rolling Stone", "entertainment", "https://rollingstone.com/feed", "US"),
    Source("deadline", "Deadline", "entertainment", "https://deadline.com/feed", "US"),
)

_BY_ID = {s.id: s for s in SOURCES}
_BY_CATEGORY: dict[str, list[Source]] = {}
for _s in SOURCES:
    _BY_CATEGORY.setdefault(_s.category, []).append(_s)


def get_source_by_id(source_id: str) -> Optional[Source]:
    return _BY_ID.get(source_id)


def get_sources_by_category(category: str) -> list[Source]:
    return list(_BY_CATEGORY.get(category, []))
