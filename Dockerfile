FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Upgrade pip and install required packages in one step to optimize layers
RUN pip install --upgrade pip \
    && pip install --prefer-binary --timeout=120 -r requirements.txt


# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run Streamlit when the container launches
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
