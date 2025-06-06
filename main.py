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
# from dotenv import load_dotenv
# load_dotenv()


app = FastAPI(title="TDS Virtual TA", description="Virtual Teaching Assistant for Tools in Data Science")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
                    course_content.append(json.loads(line.strip()))
        print(f"Loaded {len(course_content)} course content items")
    except FileNotFoundError:
        print("CourseContentData.jsonl not found")
    
    # Load discourse posts
    try:
        with open('DiscourseData.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    discourse_posts.append(json.loads(line.strip()))
        print(f"Loaded {len(discourse_posts)} discourse posts")
    except FileNotFoundError:
        print("DiscourseData.jsonl not found")

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings using AI Proxy"""
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
            timeout=30
        )
        response.raise_for_status()
        
        data = response.json()
        embeddings = [item["embedding"] for item in data["data"]]
        
        # Print cost information
        if "cost" in response.headers:
            print(f"Embedding cost: ${response.headers['cost']}")
        
        # Cache the result
        embeddings_cache[cache_key] = embeddings
        return embeddings
    
    except Exception as e:
        print(f"Error getting embeddings: {e}")
        return []

def preprocess_question(question: str) -> str:
    """Clean and preprocess the question"""
    # Remove extra whitespace
    question = re.sub(r'\s+', ' ', question.strip())
    return question.lower()

def find_relevant_content(question: str, top_k: int = 5) -> List[Dict]:
    """Find relevant content using AI Proxy embeddings"""
    if not aiproxy_token:
        return []
    
    # Combine all content for search
    all_content = []
    
    # Add course content
    for content in course_content:
        all_content.append({
            'text': content['content'],
            'url': content['url'],
            'type': 'course',
            'source': 'Course Content'
        })
    
    # Add discourse posts
    for post in discourse_posts:
        all_content.append({
            'text': post['content'],
            'url': post['url'],
            'type': 'discourse',
            'source': f"@{post['username']}",
            'username': post.get('username', 'Unknown'),
            'created_at': post.get('created_at', '')
        })
    
    if not all_content:
        return []
    
    try:
        # Get embeddings for question and all content
        all_texts = [question] + [item['text'] for item in all_content]
        embeddings = get_embeddings(all_texts)
        
        if not embeddings or len(embeddings) < 2:
            return []
        
        # Calculate similarities
        question_embedding = np.array(embeddings[0]).reshape(1, -1)
        content_embeddings = np.array(embeddings[1:])
        
        similarities = cosine_similarity(question_embedding, content_embeddings)[0]
        
        # Get top-k most similar content
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        relevant_content = []
        for idx in top_indices:
            if similarities[idx] > 0.3:  # Threshold for relevance
                relevant_content.append({
                    **all_content[idx],
                    'similarity': float(similarities[idx])
                })
        
        return relevant_content
    
    except Exception as e:
        print(f"Error in semantic search: {e}")
        return []

def match_question_pattern(question: str) -> Optional[str]:
    """Match question against known patterns"""
    question_lower = question.lower()
    
    for pattern_name, pattern_data in QUESTION_PATTERNS.items():
        if any(keyword in question_lower for keyword in pattern_data['keywords']):
            return pattern_data['response']
    
    return None

def generate_answer_with_aiproxy(question: str, relevant_content: List[Dict], image_description: str = None) -> str:
    """Generate answer using AI Proxy"""
    if not aiproxy_token:
        return generate_fallback_answer(question, relevant_content)
    
    # Prepare context from relevant content
    context = ""
    for i, content in enumerate(relevant_content[:3]):  # Top 3 most relevant
        context += f"\n--- Source {i+1}: {content['source']} ---\n{content['text'][:500]}...\n"
    
    # Prepare the prompt
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
    
    try:
        response = requests.post(
            f"{AIPROXY_BASE_URL}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        
        data = response.json()
        answer = data["choices"][0]["message"]["content"].strip()
        
        # Print cost information
        if "cost" in response.headers:
            print(f"Chat completion cost: ${response.headers['cost']}")
        
        return answer
    
    except Exception as e:
        print(f"Error with AI Proxy chat completion: {e}")
        return generate_fallback_answer(question, relevant_content)

def generate_fallback_answer(question: str, relevant_content: List[Dict]) -> str:
    """Fallback answer if AI Proxy is unavailable or fails"""
    if relevant_content:
        # Return a simple answer based on the most relevant content
        top_content = relevant_content[0]
        return f"Based on the course materials and discussions: {top_content['text'][:300]}..."
    else:
        return "Sorry, I couldn't find relevant information to answer your question. Please check the course materials or ask on the forum."

def generate_answer(question: str, relevant_content: List[Dict]) -> str:
    """Generate answer based on content and patterns"""
    
    # First, check for pattern matches
    pattern_answer = match_question_pattern(question)
    if pattern_answer and not relevant_content:
        return pattern_answer
    
    # Use AI Proxy for enhanced answers if available
    if aiproxy_token and relevant_content:
        return generate_answer_with_aiproxy(question, relevant_content)
    
    # Fallback to pattern matching or simple content-based answer
    if pattern_answer:
        return pattern_answer
    
    return generate_fallback_answer(question, relevant_content)

def extract_links(relevant_content: List[Dict]) -> List[LinkResponse]:
    """Extract relevant links from content"""
    links = []
    
    for content in relevant_content[:3]:  # Top 3 most relevant
        if content['type'] == 'discourse':
            # Clean up discourse post text for link description
            text_preview = content['text'][:100].replace('\n', ' ').strip()
            if len(content['text']) > 100:
                text_preview += "..."
            
            link_text = f"{content['source']}: {text_preview}"
        else:
            link_text = "Course Content"
        
        links.append(LinkResponse(
            url=content['url'],
            text=link_text
        ))
    
    return links

@app.on_event("startup")
async def startup_event():
    """Load data on startup"""
    load_data()

@app.post("/api/", response_model=AnswerResponse)
async def answer_question(request: QuestionRequest):
    """Main API endpoint to answer student questions"""
    
    try:
        # Preprocess question
        processed_question = preprocess_question(request.question)
        
        # Find relevant content
        relevant_content = find_relevant_content(processed_question)
        
        # Generate answer
        answer = generate_answer(request.question, relevant_content)
        
        # Extract links
        links = extract_links(relevant_content)
        
        return AnswerResponse(
            answer=answer,
            links=links
        )
    
    except Exception as e:
        print(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail="Error processing your question. Please try again.")

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
