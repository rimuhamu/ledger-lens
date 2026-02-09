import os
import logging
import requests
from typing import Dict, List
from datetime import datetime, timedelta
from functools import lru_cache

logger = logging.getLogger(__name__)


class GeopoliticalRiskService:
    """
    Service to fetch and analyze geopolitical risks using external APIs.
    Supports multiple data sources for comprehensive risk assessment.
    """
    
    def __init__(self):
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.world_bank_api_enabled = True  
        
    @lru_cache(maxsize=100)
    def get_country_risks(self, country: str, region: str = None):
        """
        Fetch geopolitical risks for a specific country or region.
        
        Args:
            country: Country name or code (e.g., "Indonesia", "United States")
            region: Optional region (e.g., "Southeast Asia", "Europe")
            
        Returns:
            List of risk factors with severity and descriptions
        """
        risks = []
        
        risks.extend(self._fetch_from_news_api(country))
        risks.extend(self._fetch_from_world_bank(country))
        risks.extend(self._fetch_from_gdelt(country, region))
        
        return self._consolidate_risks(risks)
    
    def _fetch_from_news_api(self, country: str):
        """
        Fetch recent geopolitical news using NewsAPI.
        Requires NEWS_API_KEY in environment.
        """
        if not self.news_api_key:
            logger.info("NewsAPI key not configured, skipping news-based risk assessment")
            return []
        
        try:
            keywords = f"{country} AND (sanctions OR embargo OR conflict OR trade war OR regulatory OR restrictions)"
            
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": keywords,
                "apiKey": self.news_api_key,
                "language": "en",
                "sortBy": "publishedAt",
                "from": (datetime.now() - timedelta(days=30)).isoformat(),
                "pageSize": 10
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            risks = []
            if data.get("articles"):
                # Analyze headlines for risk indicators
                for article in data["articles"][:5]:  # Top 5 articles
                    title = article.get("title", "").lower()
                    description = article.get("description", "").lower()
                    
                    # Keyword-based severity assessment
                    severity = self._assess_severity_from_text(title + " " + description)
                    
                    if severity:
                        risks.append({
                            "source": "NewsAPI",
                            "name": self._extract_risk_name(title, country),
                            "severity": severity,
                            "description": article.get("description", "")[:150],
                            "date": article.get("publishedAt"),
                            "url": article.get("url")
                        })
            
            return risks
            
        except Exception as e:
            logger.warning(f"Failed to fetch news data: {e}")
            return []
    
    def _fetch_from_world_bank(self, country: str):
        """
        Fetch governance and regulatory indicators from World Bank API.
        Free API, no key required.
        """
        try:
            # World Bank Governance Indicators
            # Using Worldwide Governance Indicators (WGI) dataset
            country_code = self._get_country_code(country)
            if not country_code:
                return []
            
            indicators = {
                "PV.EST": "Political Stability",  # Political Stability and Absence of Violence
                "RQ.EST": "Regulatory Quality",    # Regulatory Quality
                "RL.EST": "Rule of Law",          # Rule of Law
                "CC.EST": "Control of Corruption" # Control of Corruption
            }
            
            risks = []
            base_url = "https://api.worldbank.org/v2/country"
            
            for indicator_code, indicator_name in indicators.items():
                try:
                    url = f"{base_url}/{country_code}/indicator/{indicator_code}"
                    params = {"format": "json", "per_page": 1, "date": "2022:2023"}
                    
                    response = requests.get(url, params=params, timeout=10)
                    response.raise_for_status()
                    data = response.json()
                    
                    if len(data) > 1 and data[1]:
                        latest = data[1][0]
                        value = latest.get("value")
                        
                        if value is not None:
                            # World Bank scores range from -2.5 (weak) to 2.5 (strong)
                            # Convert to risk severity
                            if value < -1.0:
                                severity = "HIGH"
                                risks.append({
                                    "source": "World Bank",
                                    "name": f"{indicator_name} Concerns",
                                    "severity": severity,
                                    "description": f"{indicator_name} score: {value:.2f} (weak governance indicator)",
                                    "metric": value
                                })
                            elif value < 0:
                                severity = "MED"
                                risks.append({
                                    "source": "World Bank",
                                    "name": f"{indicator_name} Challenges",
                                    "severity": severity,
                                    "description": f"{indicator_name} score: {value:.2f} (moderate concerns)",
                                    "metric": value
                                })
                    
                except Exception as e:
                    logger.debug(f"Could not fetch {indicator_name}: {e}")
                    continue
            
            return risks
            
        except Exception as e:
            logger.warning(f"Failed to fetch World Bank data: {e}")
            return []
    
    def _fetch_from_gdelt(self, country: str, region: str = None):
        """
        Fetch geopolitical events from GDELT (Global Database of Events, Language, and Tone).
        Free API for recent event analysis. Includes retry logic for rate limiting.
        """
        import time
        
        query_location = region if region else country
        
        # GDELT 2.0 Event Database query
        url = "https://api.gdeltproject.org/api/v2/doc/doc"
        params = {
            "query": f"{query_location} (sanctions OR embargo OR trade restrictions OR conflict)",
            "mode": "artlist",
            "maxrecords": 5,  # Reduced to minimize rate limiting
            "format": "json",
            "timespan": "7d"
        }
        
        # Retry with exponential backoff
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, timeout=15)
                
                # Handle rate limiting explicitly
                if response.status_code == 429:
                    wait_time = (2 ** attempt) + 1  # 2s, 3s, 5s
                    logger.info(f"GDELT rate limited, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                    
                response.raise_for_status()
                data = response.json()
                
                risks = []
                if data.get("articles"):
                    for article in data["articles"][:3]:  # Only top 3
                        title = article.get("title", "").lower()
                        
                        severity = self._assess_severity_from_text(title)
                        if severity:
                            risks.append({
                                "source": "GDELT",
                                "name": self._extract_risk_name(title, country),
                                "severity": severity,
                                "description": title[:150],
                                "date": article.get("seendate"),
                                "url": article.get("url")
                            })
                
                return risks
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429 and attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + 1
                    logger.info(f"GDELT rate limited, waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue
                logger.debug(f"GDELT API error: {e}")
                return []
            except Exception as e:
                logger.debug(f"GDELT fetch failed: {e}")
                return []
        
        logger.info("GDELT API rate limit exceeded after retries, skipping")
        return []
    
    def _assess_severity_from_text(self, text: str):
        """
        Assess risk severity based on keyword analysis.
        """
        text = text.lower()
        
        high_risk_keywords = [
            "war", "invasion", "conflict", "sanctions", "embargo", 
            "crisis", "collapse", "ban", "prohibited", "severe"
        ]
        
        med_risk_keywords = [
            "restrictions", "tensions", "dispute", "warning", 
            "concerns", "challenges", "uncertainty", "volatility"
        ]
        
        low_risk_keywords = [
            "monitoring", "watch", "caution", "potential", "possible"
        ]
        
        if any(keyword in text for keyword in high_risk_keywords):
            return "HIGH"
        elif any(keyword in text for keyword in med_risk_keywords):
            return "MED"
        elif any(keyword in text for keyword in low_risk_keywords):
            return "LOW"
        
        return None
    
    def _extract_risk_name(self, text: str, country: str):
        """
        Extract a concise risk name from article text.
        """
        text = text.lower()
        
        # Common risk patterns
        if "sanction" in text:
            return "Economic Sanctions"
        elif "trade war" in text or "tariff" in text:
            return "Trade Restrictions"
        elif "embargo" in text:
            return "Trade Embargo"
        elif "conflict" in text or "tension" in text:
            return "Geopolitical Tensions"
        elif "regulatory" in text or "regulation" in text:
            return "Regulatory Changes"
        elif "political" in text and ("unstable" in text or "crisis" in text):
            return "Political Instability"
        else:
            return "Geopolitical Risk"
    
    def _consolidate_risks(self, risks: List[Dict]):
        """
        Deduplicate and consolidate risks from multiple sources.
        """
        if not risks:
            return []
        
        # Group by risk name
        consolidated = {}
        for risk in risks:
            name = risk["name"]
            if name not in consolidated:
                consolidated[name] = risk
            else:
                # Keep the higher severity
                existing_severity = consolidated[name]["severity"]
                new_severity = risk["severity"]
                
                severity_order = {"LOW": 1, "MED": 2, "HIGH": 3}
                if severity_order.get(new_severity, 0) > severity_order.get(existing_severity, 0):
                    consolidated[name] = risk
        
        # Sort by severity (HIGH -> MED -> LOW)
        severity_order = {"HIGH": 0, "MED": 1, "LOW": 2}
        sorted_risks = sorted(
            consolidated.values(), 
            key=lambda x: severity_order.get(x["severity"], 3)
        )
        
        return sorted_risks[:4]  # Return top 4 risks
    
    def _get_country_code(self, country: str):
        """
        Convert country name to ISO 3166-1 alpha-2 code for World Bank API.
        """
        # Common country mappings
        country_codes = {
            "indonesia": "ID",
            "united states": "US",
            "usa": "US",
            "china": "CN",
            "japan": "JP",
            "singapore": "SG",
            "malaysia": "MY",
            "thailand": "TH",
            "vietnam": "VN",
            "philippines": "PH",
            "india": "IN",
            "australia": "AU",
            "south korea": "KR",
            "korea": "KR",
            "united kingdom": "GB",
            "uk": "GB",
            "germany": "DE",
            "france": "FR",
            "brazil": "BR",
            "mexico": "MX",
            "canada": "CA"
        }
        
        return country_codes.get(country.lower())
    
    def get_risk_summary(self, country: str, region: str = None):
        """
        Get a text summary of geopolitical risks for a country/region.
        """
        risks = self.get_country_risks(country, region)
        
        if not risks:
            return f"No significant geopolitical risks identified for {country} in recent data."
        
        summary_lines = [f"Geopolitical Risk Assessment for {country}:"]
        for risk in risks:
            summary_lines.append(
                f"- {risk['name']} [{risk['severity']}]: {risk.get('description', 'No details available')[:100]}"
            )
        
        return "\n".join(summary_lines)


# Singleton instance
_geopolitical_service = None

def get_geopolitical_service():
    """Get or create the geopolitical risk service singleton."""
    global _geopolitical_service
    if _geopolitical_service is None:
        _geopolitical_service = GeopoliticalRiskService()
    return _geopolitical_service