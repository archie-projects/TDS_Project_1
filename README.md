# TDS Virtual TA

A Virtual Teaching Assistant for the Tools in Data Science course at IIT Madras that automatically answers student questions based on course content and discourse discussions.

## Features

- Semantic search through course content and discourse posts
- Pattern-based question matching for common queries
- REST API with JSON responses
- Support for image attachments (base64 encoded)
- Relevant link extraction and citation

## Setup

### Prerequisites

- Python 3.8+
- pip or uv package manager

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd tds-virtual-ta
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your data files in the root directory:
   - `CourseContentData.jsonl`
   - `DiscourseData.jsonl`

4. (Optional) Set OpenAI API key for enhanced responses:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Running the Application

```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Usage

### Endpoint: POST /api/

Send a POST request with a JSON payload:

```json
{
  "question": "Should I use gpt-4o-mini which AI proxy supports, or gpt3.5 turbo?",
  "image": "base64-encoded-image-data"  // optional
}
```

### Response Format

```json
{
  "answer": "You must use `gpt-3.5-turbo-0125`, even if the AI Proxy only supports `gpt-4o-mini`. Use the OpenAI API directly for this question.",
  "links": [
    {
      "url": "https://discourse.onlinedegree.iitm.ac.in/t/ga5-question-8-clarification/155939/4",
      "text": "Use the model that's mentioned in the question."
    }
  ]
}
```

### Example cURL Request

```bash
curl "http://localhost:8000/api/" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"How do I install Python packages?\"}"
```

## Deployment

### Using Docker

1. Build the Docker image:
```bash
docker build -t tds-virtual-ta .
```

2. Run the container:
```bash
docker run -p 8000:8000 tds-virtual-ta
```

### Using Vercel

1. Install Vercel CLI:
```bash
npm i -g vercel
```

2. Deploy:
```bash
vercel --prod
```

### Using Railway/Render

1. Connect your GitHub repository
2. Set the start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
3. Deploy

## Data Format

### Course Content (CourseContentData.jsonl)
```json
{"content": "course content text", "url": "https://example.com/content"}
```

### Discourse Posts (DiscourseData.jsonl)
```json
{"id": 123, "topic_id": 456, "url": "https://discourse.example.com/post", "username": "student", "content": "post content", "created_at": "2025-01-01T00:00:00.000Z"}
```

## Architecture

The application uses:
- **FastAPI** for the REST API
- **Sentence Transformers** for semantic similarity search
- **scikit-learn** for cosine similarity computation
- **OpenAI API** (optional) for enhanced answer generation
- **Pattern matching** for common question types

## Question Types Supported

- AI model selection (GPT variants)
- Installation and setup issues
- Assignment clarifications
- Course content questions
- General troubleshooting

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For issues related to the TDS course, please use the official discourse forum. For technical issues with this application, create an issue in this repository.
# tds_project_1
