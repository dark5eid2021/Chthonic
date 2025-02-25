# Use a Python runtime as a parent image
FROM python:3.9-slim

# Install necessary packages
RUN pip install --no-cache-dir transformers torch

# Set the working directory
WORKDIR /app

# Copy the log processor script into the container
COPY argo_log_processor.py .

# Expose any necessary ports (if applicable) or simply run the script
CMD ["python", "argo_log_processor.py"]

# docker build -t my-docker-repo/argo-log-processor:latest .
# docker push my-docker-repo/argo-log-processor:latest
