import requests
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import argparse

class DiscourseScraper:
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        """
        Initialize the discourse scraper
        
        Args:
            base_url: Base URL of the discourse instance (e.g., https://discourse.onlinedegree.iitm.ac.in)
            api_key: Optional API key for authenticated requests
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({
                'Api-Key': api_key,
                'Api-Username': 'system'
            })
    
    def get_category_topics(self, category_slug: str, page: int = 0) -> List[Dict]:
        """Get topics from a specific category"""
        url = f"{self.base_url}/c/{category_slug}.json"
        params = {'page': page}
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            return data.get('topic_list', {}).get('topics', [])
        except Exception as e:
            print(f"Error fetching topics from category {category_slug}: {e}")
            return []
    
    def get_topic_posts(self, topic_id: int) -> List[Dict]:
        """Get all posts from a specific topic"""
        url = f"{self.base_url}/t/{topic_id}.json"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            
            posts = []
            for post in data.get('post_stream', {}).get('posts', []):
                posts.append({
                    'id': post['id'],
                    'topic_id': topic_id,
                    'url': f"{self.base_url}/t/{topic_id}/{post['post_number']}",
                    'username': post['username'],
                    'content': post['cooked'],  # HTML content
                    'raw_content': post.get('raw', ''),  # Raw markdown
                    'created_at': post['created_at'],
                    'updated_at': post['updated_at'],
                    'post_number': post['post_number']
                })
            
            return posts
        except Exception as e:
            print(f"Error fetching posts from topic {topic_id}: {e}")
            return []
    
    def get_latest_topics(self, limit: int = 50) -> List[Dict]:
        """Get latest topics from the discourse instance"""
        url = f"{self.base_url}/latest.json"
        params = {'limit': limit}
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            return data.get('topic_list', {}).get('topics', [])
        except Exception as e:
            print(f"Error fetching latest topics: {e}")
            return []
    
    def filter_posts_by_date(self, posts: List[Dict], start_date: str, end_date: str) -> List[Dict]:
        """Filter posts by date range"""
        start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        
        filtered_posts = []
        for post in posts:
            post_dt = datetime.fromisoformat(post['created_at'].replace('Z', '+00:00'))
            if start_dt <= post_dt <= end_dt:
                filtered_posts.append(post)
        
        return filtered_posts
    
    def scrape_discourse_data(self, 
                            start_date: str = "2025-01-01T00:00:00.000Z",
                            end_date: str = "2025-04-14T23:59:59.999Z",
                            category_slug: str = None,
                            output_file: str = "DiscourseData.jsonl") -> List[Dict]:
        """
        Scrape discourse data within a date range
        
        Args:
            start_date: Start date in ISO format
            end_date: End date in ISO format
            category_slug: Optional category to filter by
            output_file: Output file name
        """
        all_posts = []
        
        print(f"Scraping discourse data from {start_date} to {end_date}")
        
        # Get topics based on category or latest
        if category_slug:
            print(f"Fetching topics from category: {category_slug}")
            topics = self.get_category_topics(category_slug)
        else:
            print("Fetching latest topics")
            topics = self.get_latest_topics(limit=100)
        
        print(f"Found {len(topics)} topics to process")
        
        # Process each topic
        for i, topic in enumerate(topics):
            topic_id = topic['id']
            topic_title = topic['title']
            
            print(f"Processing topic {i+1}/{len(topics)}: {topic_title}")
            
            # Get posts from this topic
            posts = self.get_topic_posts(topic_id)
            
            # Filter by date range
            filtered_posts = self.filter_posts_by_date(posts, start_date, end_date)
            
            # Clean and format posts
            for post in filtered_posts:
                # Remove HTML tags from content for cleaner text
                import re
                clean_content = re.sub(r'<[^>]+>', '', post['content'])
                clean_content = re.sub(r'\n+', '\n', clean_content).strip()
                
                cleaned_post = {
                    'id': post['id'],
                    'topic_id': post['topic_id'],
                    'url': post['url'],
                    'username': post['username'],
                    'content': clean_content,
                    'created_at': post['created_at'],
                    'topic_title': topic_title
                }
                
                all_posts.append(cleaned_post)
            
            # Rate limiting
            time.sleep(0.5)
        
        print(f"Scraped {len(all_posts)} posts total")
        
        # Save to JSONL file
        with open(output_file, 'w', encoding='utf-8') as f:
            for post in all_posts:
                f.write(json.dumps(post, ensure_ascii=False) + '\n')
        
        print(f"Data saved to {output_file}")
        return all_posts

def main():
    parser = argparse.ArgumentParser(description='Scrape Discourse posts within a date range')
    parser.add_argument('--base-url', required=True, help='Base URL of the discourse instance')
    parser.add_argument('--start-date', default='2025-01-01T00:00:00.000Z', help='Start date (ISO format)')
    parser.add_argument('--end-date', default='2025-04-14T23:59:59.999Z', help='End date (ISO format)')
    parser.add_argument('--category', help='Category slug to filter by')
    parser.add_argument('--api-key', help='Discourse API key (optional)')
    parser.add_argument('--output', default='DiscourseData.jsonl', help='Output file name')
    
    args = parser.parse_args()
    
    # Initialize scraper
    scraper = DiscourseScraper(args.base_url, args.api_key)
    
    # Scrape data
    posts = scraper.scrape_discourse_data(
        start_date=args.start_date,
        end_date=args.end_date,
        category_slug=args.category,
        output_file=args.output
    )
    
    print(f"Scraping completed! Found {len(posts)} posts.")

if __name__ == "__main__":
    main()

# Example usage:
# python discourse_scraper.py --base-url https://discourse.onlinedegree.iitm.ac.in --start-date 2025-01-01T00:00:00.000Z --end-date 2025-04-14T23:59:59.999Z
