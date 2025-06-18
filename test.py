import requests
import json
import base64

# Test configuration
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test basic health check"""
    print("=== Testing Health Check ===")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_stats():
    """Test stats endpoint"""
    print("=== Testing Stats Endpoint ===")
    response = requests.get(f"{BASE_URL}/stats")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_simple_question():
    """Test simple question"""
    print("=== Testing Simple Question ===")
    payload = {
        "question": "What is Tools in Data Science about?"
    }
    response = requests.post(f"{BASE_URL}/api/", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_gpt_question():
    """Test GPT model question from requirements"""
    print("=== Testing GPT Model Question ===")
    payload = {
        "question": "Should I use gpt-4o-mini which AI proxy supports, or gpt-3.5-turbo?"
    }
    response = requests.post(f"{BASE_URL}/api/", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_question_with_image():
    """Test question with base64 image"""
    print("=== Testing Question with Image ===")
    # Create a small test image (1x1 pixel PNG)
    test_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU77gAAAAABJRU5ErkJggg=="
    
    payload = {
        "question": "What does this image show?",
        "image": test_image_b64
    }
    response = requests.post(f"{BASE_URL}/api/", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_performance():
    """Test response time"""
    print("=== Testing Performance ===")
    import time
    
    payload = {
        "question": "How do I install Python packages?"
    }
    
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/api/", json=payload)
    end_time = time.time()
    
    response_time = end_time - start_time
    print(f"Response time: {response_time:.2f} seconds")
    print(f"Status: {response.status_code}")
    print(f"Within 30s requirement: {'✅' if response_time < 30 else '❌'}")
    print()

if __name__ == "__main__":
    try:
        test_health_check()
        test_stats()
        test_simple_question()
        test_gpt_question()
        test_question_with_image()
        test_performance()
        print("=== All tests completed ===")
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the server. Make sure it's running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
