#!/usr/bin/env python3
"""
Sports Data Collection Script

Collects training data from various sports sources including:
- Wikipedia articles on sports topics
- TheSportsDB API (free sports data)
- Ball Don't Lie API (NBA data)

Usage:
    python scripts/collect_sports_data.py --output data/raw/sports_training.jsonl --samples 1000
"""

import argparse
import json
import os
import re
import time
import random
from pathlib import Path
from typing import Optional
from urllib.parse import quote

try:
    import requests
except ImportError:
    print("Installing requests...")
    import subprocess
    subprocess.check_call(["pip", "install", "requests"])
    import requests


class WikipediaSportsCollector:
    """Collect sports articles from Wikipedia."""

    BASE_URL = "https://en.wikipedia.org/w/api.php"

    # Sports-related categories and topics
    SPORTS_CATEGORIES = [
        "American_football", "National_Football_League", "NFL_teams",
        "Basketball", "National_Basketball_Association", "NBA_teams",
        "Baseball", "Major_League_Baseball", "MLB_teams",
        "Association_football", "FIFA_World_Cup", "Premier_League",
        "La_Liga", "UEFA_Champions_League", "Serie_A",
        "Ice_hockey", "National_Hockey_League", "NHL_teams",
        "Tennis", "Grand_Slam_(tennis)", "ATP_Tour", "WTA_Tour",
        "Golf", "PGA_Tour", "Major_golf_championships",
        "Boxing", "Mixed_martial_arts", "Ultimate_Fighting_Championship",
        "Cricket", "Cricket_World_Cup", "Indian_Premier_League",
        "Olympic_Games", "Summer_Olympics", "Winter_Olympics",
        "Formula_One", "NASCAR", "Indianapolis_500",
    ]

    SPORTS_SEARCH_TERMS = [
        "Super Bowl champion", "NBA Finals MVP", "World Series winner",
        "Ballon d'Or winner", "FIFA World Cup final",
        "Wimbledon champion", "US Open tennis",
        "Masters Tournament golf", "heavyweight boxing champion",
        "UFC champion", "Olympic gold medalist",
        "Stanley Cup winner", "Premier League top scorer",
        "NFL MVP", "NBA scoring leader", "MLB home run record",
        "Champions League winner", "World Cup goalkeeper",
        "tennis Grand Slam", "cricket world cup final",
        "Formula 1 world champion", "marathon world record",
    ]

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'SportsLLMDataCollector/1.0 (Educational Research)'
        })

    def get_category_members(self, category: str, limit: int = 50) -> list:
        """Get articles from a Wikipedia category."""
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'categorymembers',
            'cmtitle': f'Category:{category}',
            'cmlimit': min(limit, 500),
            'cmtype': 'page'
        }

        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=30)
            data = response.json()
            return [m['title'] for m in data.get('query', {}).get('categorymembers', [])]
        except Exception as e:
            print(f"Error fetching category {category}: {e}")
            return []

    def search_articles(self, query: str, limit: int = 20) -> list:
        """Search for articles matching a query."""
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'search',
            'srsearch': query,
            'srlimit': limit
        }

        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=30)
            data = response.json()
            return [r['title'] for r in data.get('query', {}).get('search', [])]
        except Exception as e:
            print(f"Error searching {query}: {e}")
            return []

    def get_article_content(self, title: str) -> Optional[str]:
        """Get the plain text content of a Wikipedia article."""
        params = {
            'action': 'query',
            'format': 'json',
            'titles': title,
            'prop': 'extracts',
            'explaintext': True,
            'exsectionformat': 'plain'
        }

        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=30)
            data = response.json()
            pages = data.get('query', {}).get('pages', {})

            for page_id, page_data in pages.items():
                if page_id != '-1':
                    return page_data.get('extract', '')
            return None
        except Exception as e:
            print(f"Error fetching article {title}: {e}")
            return None

    def collect(self, target_samples: int = 500) -> list:
        """Collect sports articles from Wikipedia."""
        articles = []
        seen_titles = set()

        print("Collecting Wikipedia sports articles...")

        # Collect from categories
        for category in self.SPORTS_CATEGORIES:
            if len(articles) >= target_samples:
                break

            titles = self.get_category_members(category, limit=30)
            for title in titles:
                if title in seen_titles or len(articles) >= target_samples:
                    continue

                content = self.get_article_content(title)
                if content and len(content) > 500:
                    articles.append({
                        'source': 'wikipedia',
                        'title': title,
                        'category': category,
                        'text': content[:10000]  # Limit article length
                    })
                    seen_titles.add(title)
                    print(f"  Collected: {title} ({len(articles)}/{target_samples})")

                time.sleep(0.1)  # Rate limiting

        # Collect from search terms
        for term in self.SPORTS_SEARCH_TERMS:
            if len(articles) >= target_samples:
                break

            titles = self.search_articles(term, limit=10)
            for title in titles:
                if title in seen_titles or len(articles) >= target_samples:
                    continue

                content = self.get_article_content(title)
                if content and len(content) > 500:
                    articles.append({
                        'source': 'wikipedia',
                        'title': title,
                        'search_term': term,
                        'text': content[:10000]
                    })
                    seen_titles.add(title)
                    print(f"  Collected: {title} ({len(articles)}/{target_samples})")

                time.sleep(0.1)

        return articles


class TheSportsDBCollector:
    """Collect sports data from TheSportsDB API (free tier)."""

    BASE_URL = "https://www.thesportsdb.com/api/v1/json/3"  # Free API key

    LEAGUES = [
        "4328",   # English Premier League
        "4331",   # German Bundesliga
        "4332",   # Italian Serie A
        "4335",   # Spanish La Liga
        "4334",   # French Ligue 1
        "4387",   # NBA
        "4391",   # NFL
        "4424",   # MLB
        "4380",   # NHL
        "4346",   # MLS
        "4429",   # World Cup
    ]

    def __init__(self):
        self.session = requests.Session()

    def get_league_details(self, league_id: str) -> Optional[dict]:
        """Get details about a league."""
        try:
            url = f"{self.BASE_URL}/lookupleague.php?id={league_id}"
            response = self.session.get(url, timeout=30)
            data = response.json()
            leagues = data.get('leagues', [])
            return leagues[0] if leagues else None
        except Exception as e:
            print(f"Error fetching league {league_id}: {e}")
            return None

    def get_teams_in_league(self, league_id: str) -> list:
        """Get all teams in a league."""
        try:
            url = f"{self.BASE_URL}/lookup_all_teams.php?id={league_id}"
            response = self.session.get(url, timeout=30)
            data = response.json()
            return data.get('teams', []) or []
        except Exception as e:
            print(f"Error fetching teams for league {league_id}: {e}")
            return []

    def get_player_details(self, player_name: str) -> list:
        """Search for a player by name."""
        try:
            url = f"{self.BASE_URL}/searchplayers.php?p={quote(player_name)}"
            response = self.session.get(url, timeout=30)
            data = response.json()
            return data.get('player', []) or []
        except Exception as e:
            print(f"Error fetching player {player_name}: {e}")
            return []

    def collect(self, target_samples: int = 200) -> list:
        """Collect sports data from TheSportsDB."""
        articles = []

        print("Collecting TheSportsDB data...")

        # Collect league information
        for league_id in self.LEAGUES:
            if len(articles) >= target_samples:
                break

            league = self.get_league_details(league_id)
            if league and league.get('strDescriptionEN'):
                text = f"{league.get('strLeague', 'Unknown League')}\n\n"
                text += f"Sport: {league.get('strSport', 'Unknown')}\n"
                text += f"Country: {league.get('strCountry', 'Unknown')}\n"
                text += f"First Event: {league.get('intFormedYear', 'Unknown')}\n\n"
                text += league.get('strDescriptionEN', '')

                articles.append({
                    'source': 'thesportsdb',
                    'type': 'league',
                    'title': league.get('strLeague'),
                    'text': text
                })
                print(f"  Collected league: {league.get('strLeague')} ({len(articles)}/{target_samples})")

            # Collect team information
            teams = self.get_teams_in_league(league_id)
            for team in teams[:20]:  # Limit teams per league
                if len(articles) >= target_samples:
                    break

                if team.get('strDescriptionEN'):
                    text = f"{team.get('strTeam', 'Unknown Team')}\n\n"
                    text += f"Sport: {team.get('strSport', 'Unknown')}\n"
                    text += f"League: {team.get('strLeague', 'Unknown')}\n"
                    text += f"Stadium: {team.get('strStadium', 'Unknown')}\n"
                    text += f"Location: {team.get('strStadiumLocation', 'Unknown')}\n"
                    text += f"Formed: {team.get('intFormedYear', 'Unknown')}\n\n"
                    text += team.get('strDescriptionEN', '')

                    articles.append({
                        'source': 'thesportsdb',
                        'type': 'team',
                        'title': team.get('strTeam'),
                        'text': text
                    })
                    print(f"  Collected team: {team.get('strTeam')} ({len(articles)}/{target_samples})")

            time.sleep(0.5)  # Rate limiting

        return articles


class BallDontLieCollector:
    """Collect NBA data from Ball Don't Lie API."""

    BASE_URL = "https://api.balldontlie.io/v1"

    def __init__(self, api_key: Optional[str] = None):
        self.session = requests.Session()
        if api_key:
            self.session.headers['Authorization'] = api_key

    def get_teams(self) -> list:
        """Get all NBA teams."""
        try:
            response = self.session.get(f"{self.BASE_URL}/teams", timeout=30)
            if response.status_code == 200:
                return response.json().get('data', [])
            return []
        except Exception as e:
            print(f"Error fetching NBA teams: {e}")
            return []

    def get_players(self, page: int = 1, per_page: int = 100) -> list:
        """Get NBA players."""
        try:
            params = {'page': page, 'per_page': per_page}
            response = self.session.get(f"{self.BASE_URL}/players", params=params, timeout=30)
            if response.status_code == 200:
                return response.json().get('data', [])
            return []
        except Exception as e:
            print(f"Error fetching NBA players: {e}")
            return []

    def collect(self, target_samples: int = 100) -> list:
        """Collect NBA data from Ball Don't Lie API."""
        articles = []

        print("Collecting Ball Don't Lie NBA data...")

        # Collect team information
        teams = self.get_teams()
        for team in teams:
            if len(articles) >= target_samples:
                break

            text = f"{team.get('full_name', 'Unknown Team')}\n\n"
            text += f"Abbreviation: {team.get('abbreviation', 'N/A')}\n"
            text += f"City: {team.get('city', 'Unknown')}\n"
            text += f"Conference: {team.get('conference', 'Unknown')}\n"
            text += f"Division: {team.get('division', 'Unknown')}\n"
            text += f"\nThe {team.get('full_name')} is a professional basketball team "
            text += f"based in {team.get('city', 'Unknown City')}. They compete in the "
            text += f"{team.get('conference', 'Unknown')} Conference of the NBA's "
            text += f"{team.get('division', 'Unknown')} Division."

            articles.append({
                'source': 'balldontlie',
                'type': 'team',
                'title': team.get('full_name'),
                'text': text
            })
            print(f"  Collected team: {team.get('full_name')} ({len(articles)}/{target_samples})")

        # Collect player information
        for page in range(1, 6):  # Collect 5 pages
            if len(articles) >= target_samples:
                break

            players = self.get_players(page=page, per_page=50)
            for player in players:
                if len(articles) >= target_samples:
                    break

                name = f"{player.get('first_name', '')} {player.get('last_name', '')}".strip()
                if not name:
                    continue

                text = f"{name}\n\n"
                text += f"Position: {player.get('position', 'Unknown')}\n"

                team_data = player.get('team', {})
                if team_data:
                    text += f"Team: {team_data.get('full_name', 'Unknown')}\n"

                if player.get('height'):
                    text += f"Height: {player.get('height')}\n"
                if player.get('weight'):
                    text += f"Weight: {player.get('weight')} lbs\n"

                text += f"\n{name} is a professional basketball player in the NBA"
                if team_data:
                    text += f", currently playing for the {team_data.get('full_name', 'Unknown Team')}"
                if player.get('position'):
                    text += f" as a {player.get('position')}"
                text += "."

                articles.append({
                    'source': 'balldontlie',
                    'type': 'player',
                    'title': name,
                    'text': text
                })

            time.sleep(1)  # Rate limiting

        return articles


def clean_text(text: str) -> str:
    """Clean and normalize text for training."""
    if not text:
        return ""

    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\t+', ' ', text)

    # Remove citation markers like [1], [2], etc.
    text = re.sub(r'\[\d+\]', '', text)

    # Remove "See also", "References", "External links" sections
    sections_to_remove = [
        r'\n== See also ==.*',
        r'\n== References ==.*',
        r'\n== External links ==.*',
        r'\n== Notes ==.*',
        r'\n== Bibliography ==.*',
        r'\n== Further reading ==.*',
    ]
    for pattern in sections_to_remove:
        text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)

    return text.strip()


def format_for_training(articles: list) -> list:
    """Format articles for LLM training."""
    training_data = []

    for article in articles:
        text = clean_text(article.get('text', ''))
        if not text or len(text) < 100:
            continue

        # Create training sample
        sample = {
            'text': text,
            'source': article.get('source', 'unknown'),
            'title': article.get('title', ''),
        }

        # Add metadata if available
        if article.get('category'):
            sample['category'] = article['category']
        if article.get('type'):
            sample['type'] = article['type']

        training_data.append(sample)

    return training_data


def main():
    parser = argparse.ArgumentParser(description='Collect sports training data')
    parser.add_argument('--output', type=str, default='data/raw/sports_training.jsonl',
                        help='Output file path')
    parser.add_argument('--samples', type=int, default=1000,
                        help='Target number of samples to collect')
    parser.add_argument('--wikipedia-samples', type=int, default=500,
                        help='Number of Wikipedia samples')
    parser.add_argument('--sportsdb-samples', type=int, default=300,
                        help='Number of TheSportsDB samples')
    parser.add_argument('--nba-samples', type=int, default=200,
                        help='Number of NBA samples from Ball Don\'t Lie')
    parser.add_argument('--balldontlie-key', type=str, default=None,
                        help='Ball Don\'t Lie API key (optional)')

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_articles = []

    # Collect from Wikipedia
    print("\n" + "="*60)
    print("WIKIPEDIA COLLECTION")
    print("="*60)
    wiki_collector = WikipediaSportsCollector()
    wiki_articles = wiki_collector.collect(target_samples=args.wikipedia_samples)
    all_articles.extend(wiki_articles)
    print(f"Collected {len(wiki_articles)} Wikipedia articles")

    # Collect from TheSportsDB
    print("\n" + "="*60)
    print("THESPORTSDB COLLECTION")
    print("="*60)
    sportsdb_collector = TheSportsDBCollector()
    sportsdb_articles = sportsdb_collector.collect(target_samples=args.sportsdb_samples)
    all_articles.extend(sportsdb_articles)
    print(f"Collected {len(sportsdb_articles)} TheSportsDB articles")

    # Collect from Ball Don't Lie (NBA)
    print("\n" + "="*60)
    print("BALL DON'T LIE (NBA) COLLECTION")
    print("="*60)
    nba_collector = BallDontLieCollector(api_key=args.balldontlie_key)
    nba_articles = nba_collector.collect(target_samples=args.nba_samples)
    all_articles.extend(nba_articles)
    print(f"Collected {len(nba_articles)} NBA articles")

    # Format for training
    print("\n" + "="*60)
    print("FORMATTING FOR TRAINING")
    print("="*60)
    training_data = format_for_training(all_articles)

    # Shuffle the data
    random.shuffle(training_data)

    # Save to JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\nTotal samples collected: {len(training_data)}")
    print(f"Data saved to: {output_path}")

    # Print summary
    sources = {}
    for item in training_data:
        source = item.get('source', 'unknown')
        sources[source] = sources.get(source, 0) + 1

    print("\nSamples by source:")
    for source, count in sorted(sources.items()):
        print(f"  {source}: {count}")

    # Calculate total text size
    total_chars = sum(len(item.get('text', '')) for item in training_data)
    total_words = sum(len(item.get('text', '').split()) for item in training_data)
    print(f"\nTotal characters: {total_chars:,}")
    print(f"Total words: {total_words:,}")
    print(f"Average words per sample: {total_words // len(training_data) if training_data else 0:,}")


if __name__ == '__main__':
    main()
