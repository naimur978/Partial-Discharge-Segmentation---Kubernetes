#!/bin/bash
set -e

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for required commands
for cmd in docker kubectl; do
    if ! command_exists "$cmd"; then
        echo "Error: $cmd is required but not installed."
        exit 1
    fi
done

# Check if Kubernetes is running
if ! kubectl cluster-info &>/dev/null; then
    echo "Error: Kubernetes cluster is not running. Please start your Kubernetes cluster (e.g., Docker Desktop Kubernetes or Minikube)"
    exit 1
fi

echo "Cleaning up any existing deployment..."
kubectl delete deployment pd-inference --ignore-not-found=true
kubectl delete service pd-inference-service --ignore-not-found=true

echo "Building Docker image..."
docker build -t pd-api:latest -f docker/Dockerfile .

echo "Applying Kubernetes manifests..."
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml

echo "Waiting for deployment to be ready..."
if ! kubectl wait --for=condition=available --timeout=300s deployment/pd-inference; then
    echo "Error: Deployment did not become ready within 5 minutes"
    echo "Checking pod status:"
    kubectl get pods -l app=pd-inference
    kubectl describe pods -l app=pd-inference
    exit 1
fi

echo "Setting up port forwarding..."
# Kill any existing port-forward on port 30080
lsof -ti:30080 | xargs kill -9 2>/dev/null || true
# Start port forwarding in the background
kubectl port-forward service/pd-inference-service 30080:80 &
PORT_FORWARD_PID=$!

# Give port forwarding a moment to start
sleep 2

echo "Deployment complete! The service is available at http://localhost:30080"
echo "Testing API health..."
if curl -s -f -X GET "http://localhost:30080/healthz" > /dev/null; then
    echo "✅ API is healthy and responding"
else
    echo "⚠️  Warning: API health check failed. The service might need more time to initialize."
fi

echo
echo "Available endpoints:"
echo "- Health check: curl -X GET http://localhost:30080/healthz"
echo "- Test endpoint: curl -X GET http://localhost:30080/test-text"
echo
echo "Useful commands:"
echo "- View logs: kubectl logs -l app=pd-inference -f"
echo "- Delete deployment: kubectl delete -f kubernetes/deployment.yaml -f kubernetes/service.yaml"
echo "- Stop port forwarding: kill $PORT_FORWARD_PID"
