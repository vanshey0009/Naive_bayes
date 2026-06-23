
# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
# (We'll create a requirements.txt next)
RUN pip install --no-cache-dir pandas scikit-learn

# Define environment variable
ENV PYTHONUNBUFFERED 1

# Run the spam_detector.py script
CMD ["python", "spam_detector.py"]
