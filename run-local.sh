#!/bin/bash
set -e

echo "Starting PD Inference API with Docker Compose..."
docker-compose up -d

echo "API is now running at http://localhost:8000"
echo "Test the API with: curl -X GET http://localhost:8000/healthz"
echo "For the /test-text endpoint: curl -X GET http://localhost:8000/test-text"
echo ""
echo "To view logs: docker-compose logs -f"
echo "To stop: docker-compose down"
