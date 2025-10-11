# src/scrapers/__init__.py

from .base_scraper import BaseScraper
from .transfermarkt_scraper import TransfermarktScraper
from .fbref_scraper import FBRefScraper
from .odds_scraper import OddsScraper

__all__ = [
    "BaseScraper",
    "TransfermarktScraper",
    "FBRefScraper",
    "OddsScraper",
]
