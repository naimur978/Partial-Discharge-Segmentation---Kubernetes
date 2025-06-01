#!/bin/bash
set -e

echo "Building Docker image..."
docker build -t pd-api:latest -f docker/Dockerfile .

echo "Applying Kubernetes manifests..."
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml

echo "Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/pd-inference

echo "Deployment complete! The service is available at http://localhost:30080"
echo "Test the API with: curl -X GET http://localhost:30080/healthz"
echo "For the /test-text endpoint: curl -X GET http://localhost:30080/test-text"
