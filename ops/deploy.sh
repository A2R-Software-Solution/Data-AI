#!/bin/bash
# ===========================================
# A2R RAG API - Manual Deployment Script
# Updated for us-east4 region
# ===========================================

set -e  # Exit on any error

# Configuration
PROJECT_ID="a2r-ragbot"
REGION="us-east4"
SERVICE_NAME="rag-api"
REPOSITORY="rag-repo"
IMAGE_TAG=$(git rev-parse --short HEAD)

echo "🚀 Starting deployment for A2R RAG API"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Image Tag: $IMAGE_TAG"

# Check if required tools are installed
command -v gcloud >/dev/null 2>&1 || { echo "❌ gcloud CLI is required but not installed. Aborting." >&2; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "❌ Docker is required but not installed. Aborting." >&2; exit 1; }

# Set the project
echo "📋 Setting Google Cloud project..."
gcloud config set project $PROJECT_ID

# Build the image
echo "🔨 Building Docker image..."
docker build -t $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$SERVICE_NAME:$IMAGE_TAG -f ops/Dockerfile .

# Push the image
echo "📤 Pushing image to Artifact Registry..."
docker push $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$SERVICE_NAME:$IMAGE_TAG

# Deploy to Cloud Run
echo "🚢 Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image=$REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$SERVICE_NAME:$IMAGE_TAG \
  --region=$REGION \
  --service-account=rag-api-sa@$PROJECT_ID.iam.gserviceaccount.com \
  --vpc-connector=rag-connector \
  --vpc-egress=all-traffic \
  --min-instances=1 \
  --max-instances=20 \
  --cpu=2 \
  --memory=4Gi \
  --timeout=300 \
  --concurrency=80 \
  --set-secrets=MONGO_URI=MONGO_URI:latest \
  --set-secrets=LANGCHAIN_API_KEY=LANGSMITH_API_KEY:latest \
  --set-env-vars=MONGO_DB=rag_db,MONGO_COLLECTION=documents,VECTOR_INDEX=rag-chatbot-index \
  --set-env-vars=EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
  --set-env-vars=OLLAMA_BASE_URL=http://localhost:11434,OLLAMA_MODEL=mistral \
  --set-env-vars=LANGCHAIN_TRACING_V2=true,LANGCHAIN_PROJECT=A2R-RAG \
  --set-env-vars=LOG_LEVEL=INFO,API_VERSION=1.0.0,ENVIRONMENT=production \
  --ingress=internal-and-cloud-load-balancing \
  --allow-unauthenticated \
  --port=8080

# Get the service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format='value(status.url)')

echo "✅ Deployment completed successfully!"
echo "🌐 Service URL: $SERVICE_URL"
echo "🔍 Health check: $SERVICE_URL/healthz"
echo "📊 Detailed health: $SERVICE_URL/health/detailed"

# Test the deployment
echo "🧪 Testing deployment..."
if curl -f -s "$SERVICE_URL/healthz" > /dev/null; then
    echo "✅ Health check passed!"
else
    echo "❌ Health check failed. Check the logs:"
    echo "gcloud logs read --project=$PROJECT_ID --service=$SERVICE_NAME"
fi

echo "🎉 Deployment script completed!"
echo ""
echo "Next steps:"
echo "1. Set up Load Balancer with Cloud Armor"
echo "2. Configure custom domain"
echo "3. Set up monitoring alerts"