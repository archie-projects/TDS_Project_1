import json
import base64
import os
import re
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from PIL import Image
import pytesseract
import io

load_dotenv()

app = FastAPI(title="TDS Virtual TA", description="Virtual Teaching Assistant for Tools in Data Science")

from fastapi.middleware.cors import CORSMiddleware

# Enhanced CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://exam.sanand.workers.dev",
        "http://localhost:3000",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=[
        "Accept",
        "Accept-Language",
        "Content-Language",
        "Content-Type",
        "Authorization",
        "X-Requested-With",
        "Origin",
        "Access-Control-Request-Method",
        "Access-Control-Request-Headers",
    ],
    expose_headers=["*"],
    max_age=86400,
)

# Request/Response Models
class QuestionRequest(BaseModel):
    question: str
    image: Optional[str] = None

class LinkResponse(BaseModel):
    url: str
    text: str

class AnswerResponse(BaseModel):
    answer: str
    links: List[LinkResponse]

# Global variables
course_content = []
discourse_posts = []
aiproxy_token = None
embeddings_cache = {}
executor = ThreadPoolExecutor(max_workers=4)

# AI Proxy configuration
AIPROXY_BASE_URL = "https://aiproxy.sanand.workers.dev/openai"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# Common question patterns and responses
QUESTION_PATTERNS = {
    'gpt_model': {
        'keywords': ['gpt', 'gpt-4o-mini', 'gpt-3.5-turbo', 'model', 'ai proxy'],
        'response': "You must use `gpt-3.5-turbo-0125`, even if the AI Proxy only supports `gpt-4o-mini`. Use the OpenAI API directly for assignments that specify this model."
    },
    'installation': {
        'keywords': ['install', 'setup', 'pip', 'conda', 'error', 'import'],
        'response': "For installation issues, refer to the Development Tools section. Use `uv` for Python package management as recommended in the course."
    },
    'assignment': {
        'keywords': ['assignment', 'homework', 'submission', 'deadline', 'grade'],
        'response': "For assignment-related questions, check the course timeline and submission guidelines. Make sure to follow the exact specifications provided."
    },
    'discourse': {
        'keywords': ['discourse', 'forum', 'post', 'discussion'],
        'response': "You can find discussions and additional help on the course discourse forum. Check existing posts before creating new ones."
    }
}

def load_data():
    """Load course content and discourse data"""
    global course_content, discourse_posts, aiproxy_token
    
    # Load AI Proxy token
    aiproxy_token = os.getenv("AIPROXY_TOKEN")
    if not aiproxy_token:
        print("Warning: AIPROXY_TOKEN not found in environment variables")
    
    # Load course content
    try:
        with open('CourseContentData.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())
                    course_content.append({
                        'content': data.get('content', ''),
                        'url': data.get('url', '')
                    })
        print(f"Loaded {len(course_content)} course content items")
    except FileNotFoundError:
        print("CourseContentData.jsonl not found")
    except Exception as e:
        print(f"Error loading CourseContentData.jsonl: {e}")
    
    # Load discourse posts
    try:
        with open('DiscourseData.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())
                    discourse_posts.append({
                        'content': data.get('content', ''),
                        'url': data.get('url', ''),
                        'username': data.get('username', 'Unknown'),
                        'created_at': data.get('created_at', ''),
                        'id': data.get('id', ''),
                        'topic_id': data.get('topic_id', '')
                    })
        print(f"Loaded {len(discourse_posts)} discourse posts")
    except FileNotFoundError:
        print("DiscourseData.jsonl not found")
    except Exception as e:
        print(f"Error loading DiscourseData.jsonl: {e}")
    
    # Load additional discourse data
    try:
        with open('DiscourseData1.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())
                    discourse_posts.append({
                        'content': data.get('content', ''),
                        'url': data.get('url', ''),
                        'username': data.get('username', 'Unknown'),
                        'created_at': data.get('created_at', ''),
                        'id': data.get('id', ''),
                        'topic_id': data.get('topic_id', ''),
                        'topic_title': data.get('topic_title', ''),
                        'post_number': data.get('post_number', ''),
                        'reply_count': data.get('reply_count', 0),
                        'like_count': data.get('like_count', 0)
                    })
        print(f"Loaded additional discourse posts from DiscourseData1.jsonl")
    except FileNotFoundError:
        print("DiscourseData1.jsonl not found (optional file)")
    except Exception as e:
        print(f"Error loading DiscourseData1.jsonl: {e}")
    
    print(f"Total discourse posts loaded: {len(discourse_posts)}")

def get_embeddings_sync(texts: List[str]) -> List[List[float]]:
    """Synchronous function to get embeddings"""
    if not aiproxy_token:
        return []
    
    # Check cache first
    cache_key = str(hash(tuple(texts)))
    if cache_key in embeddings_cache:
        return embeddings_cache[cache_key]
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {aiproxy_token}"
    }
    
    payload = {
        "model": EMBEDDING_MODEL,
        "input": texts
    }
    
    try:
        response = requests.post(
            f"{AIPROXY_BASE_URL}/v1/embeddings",
            headers=headers,
            json=payload,
            timeout=10  # Reduced timeout
        )
        response.raise_for_status()
        
        data = response.json()
        embeddings = [item["embedding"] for item in data["data"]]
        
        # Cache the result
        embeddings_cache[cache_key] = embeddings
        return embeddings
    
    except Exception as e:
        print(f"Error getting embeddings: {e}")
        return []

async def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings using AI Proxy asynchronously"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, get_embeddings_sync, texts)

def preprocess_question(question: str) -> str:
    """Clean and preprocess the question"""
    question = re.sub(r'\s+', ' ', question.strip())
    return question.lower()

def simple_keyword_search(question: str, top_k: int = 5) -> List[Dict]:
    """Simple keyword-based search as fallback"""
    question_words = set(question.lower().split())
    results = []
    
    # Search course content
    for content in course_content:
        content_words = set(content['content'].lower().split())
        common_words = question_words.intersection(content_words)
        if len(common_words) > 0:
            score = len(common_words) / len(question_words)
            results.append({
                'text': content['content'],
                'url': content['url'],
                'type': 'course',
                'source': 'Course Content',
                'similarity': score
            })
    
    # Search discourse posts
    for post in discourse_posts:
        content_words = set(post['content'].lower().split())
        common_words = question_words.intersection(content_words)
        if len(common_words) > 0:
            score = len(common_words) / len(question_words)
            results.append({
                'text': post['content'],
                'url': post['url'],
                'type': 'discourse',
                'source': f"@{post['username']}",
                'username': post.get('username', 'Unknown'),
                'created_at': post.get('created_at', ''),
                'similarity': score
            })
    
    # Sort by similarity and return top results
    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results[:top_k]

async def find_relevant_content(question: str, top_k: int = 5) -> List[Dict]:
    """Find relevant content using embeddings with keyword fallback"""
    if not aiproxy_token:
        print("No AI Proxy token, using keyword search")
        return simple_keyword_search(question, top_k)
    
    # Combine all content for search (limit for performance)
    all_content = []
    
    # Add course content (limit to prevent timeout)
    for content in course_content[:200]:  # Limit for performance
        if len(content['content']) > 50:  # Only include substantial content
            all_content.append({
                'text': content['content'],
                'url': content['url'],
                'type': 'course',
                'source': 'Course Content'
            })
    
    # Add discourse posts (limit to prevent timeout)
    for post in discourse_posts[:300]:  # Limit for performance
        if len(post['content']) > 50:  # Only include substantial content
            all_content.append({
                'text': post['content'],
                'url': post['url'],
                'type': 'discourse',
                'source': f"@{post['username']}",
                'username': post.get('username', 'Unknown'),
                'created_at': post.get('created_at', '')
            })
    
    if not all_content:
        print("No content available for search")
        return []
    
    try:
        # Get embeddings for question and content
        all_texts = [question] + [item['text'][:500] for item in all_content]  # Truncate for performance
        embeddings = await get_embeddings(all_texts)
        
        if not embeddings or len(embeddings) < 2:
            print("Embeddings failed, falling back to keyword search")
            return simple_keyword_search(question, top_k)
        
        # Calculate similarities
        question_embedding = np.array(embeddings[0]).reshape(1, -1)
        content_embeddings = np.array(embeddings[1:])
        
        similarities = cosine_similarity(question_embedding, content_embeddings)[0]
        
        # Get top-k most similar content
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        relevant_content = []
        for idx in top_indices:
            if similarities[idx] > 0.2:  # Lower threshold for more results
                relevant_content.append({
                    **all_content[idx],
                    'similarity': float(similarities[idx])
                })
        
        print(f"Found {len(relevant_content)} relevant items using embeddings")
        return relevant_content
    
    except Exception as e:
        print(f"Error in semantic search: {e}, falling back to keyword search")
        return simple_keyword_search(question, top_k)

def match_question_pattern(question: str) -> Optional[str]:
    """Match question against known patterns"""
    question_lower = question.lower()
    
    for pattern_name, pattern_data in QUESTION_PATTERNS.items():
        if any(keyword in question_lower for keyword in pattern_data['keywords']):
            return pattern_data['response']
    
    return None

async def generate_answer_with_aiproxy(question: str, relevant_content: List[Dict], image_description: str = None) -> str:
    """Generate answer using AI Proxy"""
    if not aiproxy_token:
        return generate_fallback_answer(question, relevant_content)
    
    # Prepare context from relevant content
    context = ""
    for i, content in enumerate(relevant_content[:3]):
        context += f"\n--- Source {i+1}: {content['source']} ---\n{content['text'][:400]}...\n"
    
    system_prompt = """You are a helpful Teaching Assistant for the Tools in Data Science course at IIT Madras. 
Answer student questions based on the provided course content and discourse discussions.
Be concise, accurate, and helpful. If you're not sure about something, say so.
Focus on practical guidance and direct answers. Keep responses under 200 words."""
    
    user_prompt = f"""Question: {question}

Context from course materials and discussions:
{context}
"""
    
    if image_description:
        user_prompt += f"\nImage description: {image_description}"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {aiproxy_token}"
    }
    
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 300,
        "temperature": 0.3
    }
    
    def make_request():
        try:
            response = requests.post(
                f"{AIPROXY_BASE_URL}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"Error with AI Proxy chat completion: {e}")
            return None
    
    loop = asyncio.get_event_loop()
    answer = await loop.run_in_executor(executor, make_request)
    
    if answer:
        return answer
    else:
        return generate_fallback_answer(question, relevant_content)

def generate_fallback_answer(question: str, relevant_content: List[Dict]) -> str:
    """Enhanced fallback answer"""
    if relevant_content:
        # Combine information from multiple sources
        answer_parts = []
        for i, content in enumerate(relevant_content[:2]):  # Use top 2 results
            preview = content['text'][:200].replace('\n', ' ').strip()
            if len(content['text']) > 200:
                preview += "..."
            answer_parts.append(f"From {content['source']}: {preview}")
        
        return "Based on the course materials and discussions:\n\n" + "\n\n".join(answer_parts)
    else:
        return "I couldn't find specific information about your question in the course materials. Please check the course content or ask on the discourse forum for more help."

async def generate_answer(question: str, relevant_content: List[Dict]) -> str:
    """Generate answer based on content and patterns"""
    
    # First, check for pattern matches
    pattern_answer = match_question_pattern(question)
    
    # If we have relevant content, try to use AI Proxy for better answers
    if relevant_content and aiproxy_token:
        ai_answer = await generate_answer_with_aiproxy(question, relevant_content)
        # If pattern answer exists and AI answer is generic, combine them
        if pattern_answer and "Sorry" in ai_answer:
            return pattern_answer + "\n\n" + ai_answer
        return ai_answer
    
    # Use pattern answer if available
    if pattern_answer:
        return pattern_answer
    
    # Fallback to content-based answer
    return generate_fallback_answer(question, relevant_content)

def extract_links(relevant_content: List[Dict]) -> List[LinkResponse]:
    """Extract relevant links from content"""
    links = []
    
    for content in relevant_content[:3]:
        if content['type'] == 'discourse':
            text_preview = content['text'][:80].replace('\n', ' ').strip()
            if len(content['text']) > 80:
                text_preview += "..."
            link_text = f"{content['source']}: {text_preview}"
        else:
            link_text = "Course Content: " + content['text'][:60].replace('\n', ' ').strip()
            if len(content['text']) > 60:
                link_text += "..."
        
        if content['url']:  # Only add if URL exists
            links.append(LinkResponse(
                url=content['url'],
                text=link_text
            ))
    
    return links

@app.on_event("startup")
async def startup_event():
    """Load data on startup"""
    load_data()

@app.options("/api/")
async def options_api():
    return {"message": "OK"}

@app.options("/")
async def options_root():
    return {"message": "OK"}

@app.post("/api/", response_model=AnswerResponse)
async def answer_question(request: QuestionRequest):
    """Main API endpoint to answer student questions"""
    
    try:
        start_time = time.time()
        
        # Preprocess question
        processed_question = preprocess_question(request.question)
        print(f"Processing question: {request.question}")
        
        # Find relevant content
        relevant_content = await find_relevant_content(processed_question)
        print(f"Found {len(relevant_content)} relevant items")
        
        # Extract image description (if image is given)
        image_description = None
        if request.image:
            image_description = extract_text_from_base64(request.image)
            if image_description:
                print(f"OCR extracted: {image_description[:100]}...")
        
        # Generate answer (now passes image description too)
        answer = await generate_answer_with_aiproxy(request.question, relevant_content, image_description)
        
        # Extract links
        links = extract_links(relevant_content)
        
        processing_time = time.time() - start_time
        print(f"Total processing time: {processing_time:.2f} seconds")
        
        return AnswerResponse(
            answer=answer,
            links=links
        )
    
    except Exception as e:
        print(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail="Error processing your question. Please try again.")


@app.post("/", response_model=AnswerResponse)
async def answer_question_root(request: QuestionRequest):
    """Alternative API endpoint at root for compatibility"""
    return await answer_question(request)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "TDS Virtual TA is running",
        "status": "healthy",
        "version": "1.0"
    }

@app.get("/stats")
async def get_stats():
    """Get statistics about loaded data"""
    return {
        "course_content_items": len(course_content),
        "discourse_posts": len(discourse_posts),
        "aiproxy_available": aiproxy_token is not None,
        "supported_patterns": list(QUESTION_PATTERNS.keys()),
        "embedding_cache_size": len(embeddings_cache)
    }

def extract_text_from_base64(base64_str: str) -> Optional[str]:
    try:
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data))
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        print(f"Image OCR failed: {e}")
        return None

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
