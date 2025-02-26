# Use an official Python runtime as a parent image.
FROM python:3.9-slim

# Set the working directory in the container.
WORKDIR /app

# Copy the current directory contents into the container.
COPY . /app

# Install any needed packages specified in requirements.txt.
# Create a requirements.txt with at least: torch, and any other dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Expose a port if you plan to serve the model (e.g., using Flask or FastAPI).
EXPOSE 8080

# Define environment variable for Python.
ENV PYTHONUNBUFFERED=1

# Run model.py (or your server entrypoint script) when the container launches.
CMD ["python", "model.py"]

