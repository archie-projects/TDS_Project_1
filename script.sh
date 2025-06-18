#!/bin/bash

# Test 1: Basic health check
echo "=== Testing Health Check ==="
curl -X GET "http://localhost:8000/" | jq

echo -e "\n=== Testing Stats Endpoint ==="
curl -X GET "http://localhost:8000/stats" | jq

# Test 2: Simple question without image
echo -e "\n=== Testing Simple Question ==="
curl -X POST "http://localhost:8000/api/" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Tools in Data Science about?"}' | jq

# Test 3: Question about GPT models (from your requirements)
echo -e "\n=== Testing GPT Model Question ==="
curl -X POST "http://localhost:8000/api/" \
  -H "Content-Type: application/json" \
  -d '{"question": "Should I use gpt-4o-mini which AI proxy supports, or gpt-3.5-turbo?"}' | jq

# Test 4: Question with base64 image (you'll need to replace with actual base64)
echo -e "\n=== Testing Question with Image ==="
curl -X POST "http://localhost:8000/api/" \
  -H "Content-Type: application/json" \
  -d '{"question": "What does this screenshot show?", "image": "iVBORw0KGgoAAAANSU..."}' | jq

echo -e "\n=== All tests completed ==="
