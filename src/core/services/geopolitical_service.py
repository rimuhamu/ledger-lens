import os
import requests
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from src.utils.logger import get_logger
from functools import lru_cache

class GeopoliticalService:
    """
    Service to fetch real-time geopolitical data from external APIs.
    """
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.news_api_key = os.getenv("NEWS_API_KEY")
        # In a real app, we would inject API clients here
    
    def get_country_risks(self, country: str) -> List[Dict[str, Any]]:
        """
        Get geopolitical risks for a specific country.
        Aggregates data from multiple sources (simulated for now).
        """
        risks = []
        
        # 1. NewsAPI for recent conflict/political news
        news_risks = self._fetch_news_risks(country)
        risks.extend(news_risks)
        
        # 2. Simulated World Bank / GDELT data
        # In production, these would be real API calls
        if country.lower() in ["china", "taiwan", "russia", "ukraine", "israel", "iran"]:
            risks.append({
                "source": "Simulated Global Conflict Tracker",
                "name": "Regional Instability",
                "severity": "HIGH",
                "description": f"Ongoing geopolitical tensions in {country} region affecting trade and supply chains.",
                "date": datetime.now().strftime("%Y-%m-%d")
            })
            
        return risks

    def _fetch_news_risks(self, country: str) -> List[Dict[str, Any]]:
        if not self.news_api_key:
            return []
            
        try:
            url = f"https://newsapi.org/v2/top-headlines?q={country}&category=business&apiKey={self.news_api_key}"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                articles = response.json().get("articles", [])
                risks = []
                
                # Simple keyword matching for risk detection
                risk_keywords = ["sanction", "conflict", "ban", "tariff", "protest", "crisis", "shortage"]
                
                for article in articles[:3]:
                    title = article.get("title", "").lower()
                    description = article.get("description", "") or ""
                    
                    if any(keyword in title or keyword in description.lower() for keyword in risk_keywords):
                        risks.append({
                            "source": "NewsAPI",
                            "name": "Breaking News Risk",
                            "severity": "MED",
                            "description": article["title"],
                            "date": article.get("publishedAt", "")[:10]
                        })
                return risks
            return []
        except Exception as e:
            self.logger.error(f"Error fetching news: {e}")
            return []

@lru_cache()
def get_geopolitical_service() -> GeopoliticalService:
    return GeopoliticalService()
