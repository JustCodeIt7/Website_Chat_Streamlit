FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy all application files from the repository
COPY . .

# Install Python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Expose the Streamlit port
EXPOSE 8501

# Configure health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the Streamlit application
ENTRYPOINT ["streamlit", "run", "website_chat.py", "--server.port=8501", "--server.address=0.0.0.0"]
