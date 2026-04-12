FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy only requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of project
COPY . .

# Set Python path
ENV PYTHONPATH="/app"

# Expose HF port
EXPOSE 7860

# Run server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]