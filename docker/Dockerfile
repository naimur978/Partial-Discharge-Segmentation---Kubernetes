# Use an official Python runtime as the base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install curl for testing the endpoint
RUN apt-get update && apt-get install -y curl && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model, code, and test data
COPY api/app.py .
COPY api/pd_inference.py .
COPY resources/best_model.pth .
COPY resources/test_dataa.npy .

# Make port 80 available to the world outside this container
EXPOSE 80

# Command to run the FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
