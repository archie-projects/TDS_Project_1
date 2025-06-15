import requests
import json
import time
import sys
from datetime import datetime
from typing import List, Dict, Optional
import argparse
import re
import logging

# =============================================================================
# CONFIGURATION - Replace this with your own cookie value
# =============================================================================

DISCOURSE_COOKIE_T = "YOUR_COOKIE_HERE"  # 🔴 Replace this string with your actual _t cookie

BASE_URL = "https://discourse.onlinedegree.iitm.ac.in"
CATEGORY_SLUG = "courses/tds-kb/34"  # Set to None for all categories
START_DATE = "2025-01-01T00:00:00.000Z"
END_DATE = "2025-04-14T23:59:59.999Z"
OUTPUT_FILE = "DiscourseData.jsonl"
MAX_TOPICS = None  # Set to a number to limit, or None for all

# =============================================================================

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedDiscourseScraper:
    def __init__(self, base_url: str, cookies: Dict[str, str]):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json, text/plain, */*',
        })
        self.session.cookies.update(cookies)
        logger.info("Session initialized with authentication.")

    def make_request(self, url: str, params: Optional[Dict] = None, retries: int = 3) -> Optional[Dict]:
        for attempt in range(retries):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                if 'application/json' in response.headers.get('Content-Type', ''):
                    return response.json()
                else:
                    logger.warning(f"Non-JSON response from {url}")
                    return None
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed ({attempt+1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
        logger.error(f"Failed to fetch {url}")
        return None

    def get_category_topics(self, category_slug: str, page: int = 0) -> List[Dict]:
        url = f"{self.base_url}/c/{category_slug}.json"
        data = self.make_request(url, params={'page': page})
        return data.get('topic_list', {}).get('topics', []) if data else []

    def get_all_category_topics(self, category_slug: str, max_pages: int = 10) -> List[Dict]:
        all_topics, page = [], 0
        while page < max_pages:
            topics = self.get_category_topics(category_slug, page)
            if not topics:
                break
            all_topics.extend(topics)
            page += 1
            time.sleep(1)
        return all_topics

    def get_latest_topics(self, limit: int = 50) -> List[Dict]:
        url = f"{self.base_url}/latest.json"
        data = self.make_request(url, params={'limit': limit})
        return data.get('topic_list', {}).get('topics', []) if data else []

    def get_topic_posts(self, topic_id: int) -> List[Dict]:
        url = f"{self.base_url}/t/{topic_id}.json"
        data = self.make_request(url)
        posts = []
        if not data:
            return posts
        title = data.get('title', 'Untitled')
        for post in data.get('post_stream', {}).get('posts', []):
            posts.append({
                'id': post['id'],
                'topic_id': topic_id,
                'topic_title': title,
                'url': f"{self.base_url}/t/{topic_id}/{post['post_number']}",
                'username': post['username'],
                'content': post['cooked'],
                'raw_content': post.get('raw', ''),
                'created_at': post['created_at'],
                'post_number': post['post_number'],
                'reply_count': post.get('reply_count', 0),
                'like_count': post.get('actions_summary', [{}])[0].get('count', 0)
            })
        return posts

    def clean_html(self, html: str) -> str:
        text = re.sub(r'<[^>]+>', '', html)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def filter_by_date(self, posts: List[Dict], start: str, end: str) -> List[Dict]:
        try:
            start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
        except ValueError:
            logger.error("Invalid date format")
            return posts
        result = []
        for post in posts:
            try:
                created = datetime.fromisoformat(post['created_at'].replace("Z", "+00:00"))
                if start_dt <= created <= end_dt:
                    result.append(post)
            except Exception:
                continue
        return result

    def scrape_discourse_data(self, start_date: str, end_date: str, category_slug: str, output_file: str, max_topics: Optional[int] = None) -> List[Dict]:
        topics = self.get_all_category_topics(category_slug) if category_slug else self.get_latest_topics(limit=200)
        if max_topics:
            topics = topics[:max_topics]
        logger.info(f"Processing {len(topics)} topics")

        all_posts = []
        for topic in topics:
            topic_id = topic['id']
            posts = self.get_topic_posts(topic_id)
            posts = self.filter_by_date(posts, start_date, end_date)
            for post in posts:
                content = self.clean_html(post['content'])
                if not content.strip():
                    continue
                post['content'] = content
                all_posts.append(post)
            time.sleep(0.5)
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for post in all_posts:
                    f.write(json.dumps(post, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Failed to write output: {e}")
        return all_posts


def quick_run():
    if DISCOURSE_COOKIE_T == "YOUR_COOKIE_HERE":
        print("❌ Please set DISCOURSE_COOKIE_T to your actual '_t' cookie.")
        return
    cookies = {"_t": DISCOURSE_COOKIE_T}
    scraper = EnhancedDiscourseScraper(BASE_URL, cookies)
    print("🚀 Starting scraper...")
    posts = scraper.scrape_discourse_data(
        start_date=START_DATE,
        end_date=END_DATE,
        category_slug=CATEGORY_SLUG,
        output_file=OUTPUT_FILE,
        max_topics=MAX_TOPICS
    )
    print(f"✅ Finished. {len(posts)} posts saved to {OUTPUT_FILE}")

def main():
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument('--base-url', required=True)
        parser.add_argument('--cookie-token', required=True)
        parser.add_argument('--start-date', required=True)
        parser.add_argument('--end-date', required=True)
        parser.add_argument('--category', required=False)
        parser.add_argument('--output', required=True)
        parser.add_argument('--max-topics', type=int, default=None)
        args = parser.parse_args()

        cookies = {"_t": args.cookie_token}
        scraper = EnhancedDiscourseScraper(args.base_url, cookies)
        scraper.scrape_discourse_data(
            start_date=args.start_date,
            end_date=args.end_date,
            category_slug=args.category,
            output_file=args.output,
            max_topics=args.max_topics
        )
    else:
        quick_run()

if __name__ == "__main__":
    main()
