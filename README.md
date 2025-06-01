# PD Inference API

Hi there! I've created this API for Partial Discharge (PD) signal analysis and inference. This service processes high-frequency signal data using my PyTorch UNet1D model to detect anomalies that might indicate equipment failure in high voltage systems.

## Why I Built This

As someone working with high voltage equipment diagnostics, I needed a reliable way to detect partial discharges - those tiny electrical sparks that can indicate insulation problems before they cause catastrophic failures. This API provides both visual and text-based output to help identify these issues early.

## Why Just the Model Weights?

I've chosen to save just the model weights (`best_model.pth`) rather than the entire model because:

1. It's much more efficient - the weights file is ~48MB while a full model save would be >150MB
2. It keeps the Docker image smaller for faster deployments
3. The model architecture is defined in code (in `pd_inference.py`), making it easier to adapt or modify
4. It's the standard practice for PyTorch model deployment - define the architecture and load the weights

This approach gives me more flexibility while keeping the deployment footprint small.

## System Architecture & Analysis

### Client Request Flow

```
+----------+                               +-------------------+
|          |   GET /                       |                   |
|          +------------------------------>+ API Root          |
|          |                               |                   |
|          |   GET /healthz                +-------------------+
|          +------------------------------>+                   |
|          |                               | Health Check      |
|          |   GET /test-text              |                   |
|          +------------------------------>+-------------------+
|          |                               |                   |
| Client   |                               | Text Analysis     +--+
|          |   GET /test                   |                   |  |
|          +------------------------------>+-------------------+  |  +--------------+
|          |                               |                   |  |  |              |
|          |                               | Visual Analysis   +--+->+ Load Model   +--+
|          |                               |                   |     |              |  |
|          |                               +-------------------+     +--------------+  |
|          |                                                                          |
|          |                                                                          |
|          |              +---------------------------------------------------+       |
|          |              |                                                   |       |
|          | <------------+ JSON Response                                     |       |
|          |              |                                                   |       |
|          |              +---------------------------------------------------+       |
|          |                                                                          |
|          |              +---------------------------------------------------+       |
|          | <------------+ HTML/Image Response                               |       |
|          |              |                                                   |       |
|          |              +---------------------------------------------------+       |
|          |                                                                          |
+----------+                                                                          |
                                          +------------------+                         |
                                          |                  | <---------------------+
                                          | PyTorch UNet1D   |
                                          |                  |
                                          +------------------+
                                                   |
                                                   v
                                          +------------------+
                                          |                  |
                                          | best_model.pth   |
                                          |                  |
                                          +------------------+
```

### Container Architecture

```
+--------------------------------------------------------------+
|  Kubernetes Cluster                                          |
|                                                              |
|  +------------------------+      +------------------------+  |
|  |  Pod 1                 |      |  Pod 2                 |  |
|  |                        |      |                        |  |
|  |  +------------------+  |      |  +------------------+  |  |
|  |  | Container:       |  |      |  | Container:       |  |  |
|  |  | pd-inference     |  |      |  | pd-inference     |  |  |
|  |  |                  |  |      |  |                  |  |  |
|  |  | +-------------+  |  |      |  | +-------------+  |  |  |
|  |  | | Volume:     |  |  |      |  | | Volume:     |  |  |  |
|  |  | | model-cache |  |  |      |  | | model-cache |  |  |  |
|  |  | +-------------+  |  |      |  | +-------------+  |  |  |
|  |  |                  |  |      |  |                  |  |  |
|  |  +------------------+  |      |  +------------------+  |  |
|  |         |              |      |         |              |  |
|  |         v              |      |         v              |  |
|  |       Port 80          |      |       Port 80          |  |
|  |                        |      |                        |  |
|  +------------------------+      +------------------------+  |
|     ^                                  ^                     |
|     |                                  |                     |
|     +----------------------------------+                     |
|                      |                                       |
|  +------------------v-------------------+                    |
|  |                                      |                    |
|  |  Service: pd-inference-service       |                    |
|  |                                      |                    |
|  +------------------+-------------------+                    |
|                     |                                        |
|                     v                                        |
|                NodePort: 30080                               |
|                                                              |
+--------------------------------------------------------------+
                     ^
                     |
                     |
                     |
            +--------+--------+
            |                 |
            |     Client      |
            |                 |
            +-----------------+
```

### Resource Allocation

| Component | CPU Request | CPU Limit | Memory Request | Memory Limit |
|-----------|------------|-----------|---------------|--------------|
| API Container | 1 CPU | 2 CPU | 2Gi | 4Gi |
| Model Cache Volume | - | - | - | 1Gi |

The resource allocation is designed to:
- Handle computation-intensive PyTorch inference operations
- Provide enough memory for processing large signal arrays
- Ensure model loading and caching is efficient
- Allow for parallel request processing with multiple worker threads

### Scaling Strategy

The application uses a multi-layered scaling approach:

1. **Container-level parallelism**
   - Configured worker threads: 4 (via WORKER_THREADS)
   - PyTorch thread control: 4 (via TORCH_NUM_THREADS)
   - OpenMP threads: 4 (via OMP_NUM_THREADS)
   - Maximum batch size: 8 (via MAX_BATCH_SIZE)

2. **Kubernetes-level scaling**
   - Static horizontal scaling with 2 replica pods
   - Anti-affinity rules to distribute across nodes
   - RollingUpdate strategy for zero-downtime deployments
   - Could be extended with Horizontal Pod Autoscaler (HPA)

3. **Performance-optimized configurations**
   - Memory-backed volume for model cache
   - Appropriate health probes with timing adjustments
   - Startup probe allowing for longer initialization

### Docker Container Performance

| Container Name | CPU % | Memory Usage | Network I/O | Disk I/O |
|---------------|-------|-------------|------------|----------|
| pd-pd-inference-api-1 | 0.24% | 244.2MiB / 7.65GB | 3.44kB/1.67kB | 1.43MB/815kB |
| pd-inference (k8s pod 1) | 0.28% | 275.7MiB / 4GB | minimal | 52.5MB/831kB |
| pd-inference (k8s pod 2) | 0.26% | 243.9MiB / 4GB | minimal | 9.8MB/815kB |

Key observations:
- Low CPU utilization during idle periods (~0.25%)
- CPU usage spikes to ~30-50% during inference (not shown in snapshot)
- Memory usage is stable at ~240-275MB, well within allocation limits
- Minimal network and disk I/O during normal operation

### Docker Images

| Image Name | Tag | Size | Created |
|------------|-----|------|---------|
| pd-pd-inference-api | latest | 1.61GB | 2025-06-01 19:09:00 +0200 |
| pd-api | latest | 1.61GB | 2025-06-01 19:09:00 +0200 |
| pd-inference | latest | 1.61GB | 2025-06-01 18:25:50 +0200 |

Container size optimization opportunities:
- Consider multi-stage Docker builds
- Remove unnecessary Python packages
- Use lighter base images like python:3.10-slim-bullseye

## Project Structure

```
├── api/                  # API code
│   ├── app.py            # FastAPI application
│   └── pd_inference.py   # Inference logic
├── docker/               # Docker-related files
│   └── Dockerfile        # Dockerfile for building the container
├── kubernetes/           # Kubernetes manifests
│   ├── deployment.yaml   # Kubernetes deployment configuration
│   └── service.yaml      # Kubernetes service configuration
├── resources/            # Resources needed by the application
│   ├── best_model.pth    # Trained model weights
│   └── test_dataa.npy    # Test data
├── docs/                 # Documentation files
├── deploy.sh             # Script to build and deploy to Kubernetes
├── run-local.sh          # Script to run locally with Docker Compose
├── docker-compose.yml    # Docker Compose configuration
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## Prerequisites

- Docker Desktop with Kubernetes enabled (for Kubernetes deployment)
- kubectl command-line tool configured for your Kubernetes cluster
- Docker and Docker Compose for local development
- At least 4GB of available memory for the container

## Running Locally with Docker Compose

### Option 1: Using the provided script

The easiest way to run the application locally:

```bash
# Make the script executable (if needed)
chmod +x run-local.sh

# Run the script
./run-local.sh
```

### Option 2: Manual Docker Compose

```bash
# Build and start the container
docker-compose up --build -d

# Check the logs
docker-compose logs -f

# Stop the container when done
docker-compose down
```

The API will be available at http://localhost:8000

### Testing the local deployment

```bash
# Test the health check endpoint
curl -X GET http://localhost:8000/healthz

# Test the root endpoint
curl -X GET http://localhost:8000/

# Test the text-based analysis endpoint
curl -X GET http://localhost:8000/test-text

# Test the visual analysis endpoint (view in a browser)
open http://localhost:8000/test
```

## Running in Kubernetes

### Option 1: Using the deploy script (Recommended)

The easiest way to deploy to Kubernetes:

```bash
# Make the script executable (if needed)
chmod +x deploy.sh

# Run the deployment script
./deploy.sh
```

### Option 2: Manual deployment

Step-by-step manual deployment:

1. Build the Docker image:

```bash
docker build -t pd-api:latest -f docker/Dockerfile .
```

2. Verify the image was created:

```bash
docker images | grep pd-api
```

3. Apply the Kubernetes manifests:

```bash
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
```

4. Check the deployment status:

```bash
kubectl get deployments
kubectl get pods
kubectl get services
```

5. Wait for the deployment to be ready:

```bash
kubectl wait --for=condition=available --timeout=300s deployment/pd-inference
```

6. Since Kubernetes in Docker Desktop may not expose NodePort services directly, set up port-forwarding:

```bash
# Forward the service port to your local machine
kubectl port-forward service/pd-inference-service 8080:80
```

7. The service will be available at:

```
http://localhost:8080
```

### Testing the Kubernetes deployment

```bash
# Test the health check endpoint
curl -X GET http://localhost:8080/healthz

# Test the root endpoint
curl -X GET http://localhost:8080/

# Test the text-based analysis endpoint
curl -X GET http://localhost:8080/test-text

# View logs from the pods
kubectl logs -l app=pd-inference

# If you need to delete the deployment
kubectl delete -f kubernetes/deployment.yaml -f kubernetes/service.yaml
```

## API Endpoints

- **`/`** - Root endpoint (returns a welcome message)
  ```bash
  curl -X GET http://localhost:8080/
  ```

- **`/healthz`** - Health check endpoint (verifies if the model is loaded)
  ```bash
  curl -X GET http://localhost:8080/healthz
  ```

- **`/test`** - Visual test endpoint (returns HTML with visualizations)
  ```bash
  # Open in a browser
  open http://localhost:8080/test
  ```
  
  This endpoint generates visualizations like the one below, showing signal plots with detected PD regions highlighted in red:
  
  ![PD Signal Visualization](docs/pd_signal_visualization.png)
  
  I created these visualizations to help engineers quickly identify PD patterns in signals. As you can see in the image:
  
  - The top plots show the raw signal data
  - The middle plots display the filtered signal after preprocessing
  - The bottom plots highlight regions where my model has detected potential partial discharges (in red boxes)
  
  This visual representation makes it much easier to identify patterns and verify the model's detection accuracy.
  [Detailed documentation for the `/test` endpoint](docs/test-endpoint.md)

- **`/test-text`** - Text-based test endpoint (returns JSON analysis)
  ```bash
  curl -X GET http://localhost:8080/test-text
  ```
  
  This endpoint returns a detailed JSON analysis of the signal data. Here's a sample output:
  
  ```json
  {
    "total_samples_processed": 7,
    "valid_samples_found": 4,
    "results": [
      {
        "sample_index": 1,
        "num_regions_detected": 25,
        "regions": [
          {"start": 26, "end": 79, "duration": 53},
          {"start": 1574, "end": 1579, "duration": 5},
          // More regions...
        ],
        "signal_stats": {
          "max_amplitude": 0.17049315571784973,
          "mean_amplitude": 0.0004181701224297285,
          "std_amplitude": 0.005596190225332975
        }
      },
      // More samples...
    ]
  }
  ```
  
  I designed this structured output to be perfect for programmatic analysis or integration with monitoring systems. The JSON format makes it easy to process the data further or store it in a database.
  [Detailed documentation for the `/test-text` endpoint](docs/test-text-endpoint.md)

## Environment Variables

I've made the application configurable with these environment variables that you can customize in the Kubernetes deployment or docker-compose.yml:

- **`WORKER_THREADS`** - Number of worker threads for inference (default: 4)
- **`MAX_BATCH_SIZE`** - Maximum batch size for processing (default: 8)
- **`TORCH_NUM_THREADS`** - Number of PyTorch threads (default: 4)
- **`OMP_NUM_THREADS`** - Number of OpenMP threads (default: 4)

I recommend adjusting these based on your hardware capabilities. For servers with more CPUs, increasing these values can significantly improve processing speed.

## Development

### Running from Source (Without Docker)

If you want to run the application directly on your machine like I often do during development:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the FastAPI application
cd api
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Modifying the API

When I'm making changes to the API:

1. I edit the files in the `api` directory
2. With the `--reload` flag, the server automatically picks up my changes
3. For structural changes, I restart the application

I've structured the code to separate the inference logic (`pd_inference.py`) from the API endpoints (`app.py`) to make it easier to modify either component independently.

## Troubleshooting

Here are some issues I've encountered and how to solve them:

### Docker Issues

- **Image Not Found**: If you see this error, make sure you've built the Docker image with `docker build -t pd-api:latest -f docker/Dockerfile .`
- **Port Conflicts**: I've set the API to use port 8000, but if that's already in use on your machine, you can modify the port mapping in docker-compose.yml

### Kubernetes Issues

- **Pod Crashes**: I've found the most useful debugging technique is to check pod logs with `kubectl logs -l app=pd-inference`
- **Resource Limits**: The model processing can be memory-intensive. If you see pods being OOMKilled, I recommend increasing memory limits in kubernetes/deployment.yaml
- **Service Unavailable**: On macOS and Windows with Docker Desktop, NodePort services often need port-forwarding: `kubectl port-forward service/pd-inference-service 8080:80`

### API Issues

- **Model Loading Errors**: My first troubleshooting step is always to verify the model file exists in the resources directory
- **Memory Issues**: If you're processing larger signals than I anticipated, try reducing the batch size or other processing parameters through the environment variables

If you encounter any other issues, feel free to contact me. I'm always looking to improve this system!
