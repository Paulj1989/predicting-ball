# src/scrapers/base_scraper.py

import time
import random
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict
import requests


class BaseScraper(ABC):
    """Base class for all scrapers"""

    def __init__(self, min_delay: float = 5.0, max_delay: float = 15.0):
        """Scrape data for a single season"""
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.last_request_time = 0
        self.logger = logging.getLogger(self.__class__.__name__)

    def respect_rate_limit(self):
        """Enforce polite delay between requests"""
        elapsed = time.time() - self.last_request_time
        delay = random.uniform(self.min_delay, self.max_delay)

        if elapsed < delay:
            sleep_time = delay - elapsed
            self.logger.debug(f"Rate limiting: waiting {sleep_time:.1f}s")
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    @abstractmethod
    def scrape_season(self, season_end_year: int):
        """Scrape data for a single season"""
        pass

    @abstractmethod
    def scrape_multiple_seasons(self, start_year: int, end_year: int):
        """Scrape data for multiple seasons"""
        pass
