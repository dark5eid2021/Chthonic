# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and fine-tuned model
COPY . .

# Expose port 80 for FastAPI
EXPOSE 80

# Run the FastAPI app using uvicorn
CMD ["uvicorn", "model_server:app", "--host", "0.0.0.0", "--port", "80"]
