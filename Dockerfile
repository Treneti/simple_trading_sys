# Use a base image with Python
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application files to the container
COPY . .

# Expose a port for Dagit (3000 is the default port for Dagit)
EXPOSE 3000

# Define the command to run Dagit for the pipeline
CMD dagster dev -f pipeline.py -p 3000 -h 0.0.0.0