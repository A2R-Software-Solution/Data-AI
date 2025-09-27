# A2R RAG API - Production FastAPI Application

A production-ready RAG (Retrieval-Augmented Generation) API built for A2R Software Solutions. This application provides intelligent document search and question-answering capabilities using MongoDB Atlas vector search and Ollama LLM integration.

## Architecture Overview

```
Frontend (Website) → FastAPI API → MongoDB Atlas (Vector Search) → Ollama LLM
                                 ↓
                           Structured Logging & Monitoring
```

## Key Components

- **FastAPI**: Production web framework with async support
- **MongoDB Atlas**: Vector database with `$vectorSearch` capabilities
- **Ollama**: Self-hosted LLM (Mistral 7B) for response generation
- **LangChain**: LLM orchestration and prompt management
- **SymSpell**: Query spell-checking and correction
- **Google Cloud Run**: Production deployment platform

## Project Structure

```
rag-chatbot/
├── app/                          # Main application
│   ├── core/                     # Core configuration
│   │   ├── config.py            # Environment settings
│   │   └── logging.py           # Structured JSON logging
│   ├── services/                 # Business logic services
│   │   ├── retriever.py         # MongoDB Atlas vector search
│   │   ├── llm_client.py        # Ollama LLM client
│   │   └── spellcheck.py        # SymSpell integration
│   ├── routers/                  # API endpoints
│   │   ├── health.py            # Health check endpoints
│   │   └── query.py             # Main RAG endpoints
│   └── main.py                   # FastAPI application entry point
├── scripts/                      # Data pipeline utilities
│   ├── chunks_pdf.py            # PDF document processing
│   └── ingest.py                # MongoDB data ingestion
├── ops/                          # Deployment & infrastructure
│   ├── Dockerfile               # Production container
│   ├── cloudbuild.yaml          # Google Cloud Build CI/CD
│   ├── deploy.sh                # Manual deployment script
│   └── .dockerignore            # Container build exclusions
├── data/                         # Local development data
│   ├── raw/                     # Source PDF documents
│   └── processed/               # Generated chunks
├── frequency_dictionary_en_82_765.txt  # SymSpell dictionary
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
└── .gitignore                   # Git exclusions
```

## API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/query` | Main RAG endpoint - returns AI-generated answers |
| `GET` | `/healthz` | Basic health check for Cloud Run |
| `GET` | `/health/detailed` | Comprehensive service health status |

### Development Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/query/search/{query}` | Vector search only (no LLM) |
| `GET` | `/query/suggestions/{text}` | Spelling suggestions |

### Example Usage

```bash
# Main RAG query
curl -X POST "https://your-api.run.app/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is artificial intelligence?"}'

# Health check
curl "https://your-api.run.app/healthz"
```

## Environment Configuration

Required environment variables (stored in `.env` for local development, Secret Manager for production):

```bash
# MongoDB Atlas
MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/
MONGO_DB=rag_db
MONGO_COLLECTION=documents
VECTOR_INDEX=rag-chatbot-index

# Embeddings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# LLM Service
OLLAMA_BASE_URL=http://ollama-runner.internal:11434
OLLAMA_MODEL=mistral

# Application
LOG_LEVEL=INFO
API_VERSION=1.0.0
PORT=8080
```

## Data Pipeline

### 1. Document Processing
```bash
# Place PDF files in data/raw/
python scripts/chunks_pdf.py
```

### 2. Vector Embedding & Storage
```bash
# Requires MongoDB Atlas vector search index
python scripts/ingest.py
```

### 3. Vector Search Index
Create in MongoDB Atlas with these settings:
```json
{
  "fields": [{
    "type": "vector",
    "path": "embedding",
    "numDimensions": 384,
    "similarity": "cosine"
  }]
}
```

## Local Development

### Prerequisites
- Python 3.10+
- MongoDB Atlas cluster with vector search enabled
- Ollama installed locally (optional for testing)

### Setup
```bash
# Clone and setup
git clone <repository>
cd rag-chatbot
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download SymSpell dictionary
wget https://raw.githubusercontent.com/wolfgarbe/SymSpell/master/SymSpell.FrequencyDictionary/en-82_765.txt -O frequency_dictionary_en_82_765.txt

# Configure environment
cp .env.example .env
# Edit .env with your MongoDB Atlas credentials

# Run data pipeline (optional)
mkdir -p data/raw data/processed
# Add PDF files to data/raw/
python scripts/chunks_pdf.py
python scripts/ingest.py

# Start development server
uvicorn app.main:app --reload
```

### Testing
```bash
# Health check
curl http://localhost:8000/healthz

# Test query
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "test question"}'
```

## Production Deployment

### Google Cloud Infrastructure
```bash
# Enable services
gcloud services enable run.googleapis.com artifactregistry.googleapis.com

# Create Artifact Registry
gcloud artifacts repositories create rag-repo --repository-format=docker --location=us-central1

# Store secrets
echo "your-mongo-uri" | gcloud secrets create MONGO_URI --data-file=-

# Deploy via Cloud Build
gcloud builds submit --config ops/cloudbuild.yaml
```

### Security Features
- CORS restricted to production domains
- Secrets stored in Google Secret Manager
- Private VPC networking with MongoDB Atlas
- Non-root container user
- Request correlation IDs for tracing
- Structured JSON logging

### Monitoring
- Google Cloud Logging integration
- Custom health check endpoints
- LangSmith tracing integration
- Performance metrics and request timing

## Migration from FAISS

This project was migrated from a local FAISS-based implementation to production MongoDB Atlas:

### Changes Made
- **Vector Storage**: FAISS → MongoDB Atlas `$vectorSearch`
- **Architecture**: Monolithic → Microservices (separate API/frontend)
- **Security**: Wildcard CORS → Domain-restricted
- **Logging**: Basic → Structured JSON logging
- **Deployment**: Local only → Cloud Run production
- **LLM**: Local Ollama → Remote Ollama service

### Benefits
- Scalable vector search with MongoDB Atlas
- Production-grade security and monitoring
- Cloud-native deployment with auto-scaling
- Proper separation of concerns
- CI/CD pipeline integration

## Performance Characteristics

### Current Configuration
- **CPU**: 2 vCPU per Cloud Run instance
- **Memory**: 2GB RAM per instance
- **Scaling**: 1-20 instances (auto-scale)
- **Response Time**: ~100-500ms for typical queries
- **Concurrency**: 80 requests per instance

### Optimization Notes
- Vector search limited to 4 results by default
- Minimum similarity score of 0.75 for relevance
- LLM context truncated to 8000 characters
- Embedding model cached on first request

## Development Status

### Completed Features
- ✅ MongoDB Atlas vector search integration
- ✅ Ollama LLM client with error handling
- ✅ SymSpell query correction
- ✅ Production logging and monitoring
- ✅ Health check endpoints
- ✅ Docker containerization
- ✅ Cloud Build CI/CD pipeline
- ✅ Security hardening (CORS, secrets)

### Production Ready
The application is ready for production deployment with:
- Comprehensive error handling
- Security best practices
- Monitoring and observability
- Auto-scaling capabilities
- Proper secret management

## Support

For issues or questions related to this implementation, refer to the structured logs in Google Cloud Logging or check the health endpoints for service status.

---

**Built for A2R Software Solutions**  
Version: 1.0.0  
Last Updated: September 2025