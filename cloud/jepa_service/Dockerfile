FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Set up environment
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3-pip git && rm -rf /var/lib/apt/lists/*

# Install Poetry for dependency management (optional)
RUN pip3 install --no-cache-dir poetry

# Work directory
WORKDIR /app

# Copy project files
COPY . /app

# Install Python requirements (FastAPI, torch, etc.)
RUN pip3 install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 8000

# Command to run the service
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]