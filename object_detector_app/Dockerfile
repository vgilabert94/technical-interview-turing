# Use an official Python runtime as a parent image
FROM python:3.10

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install needed packages specified in requirements.txt
RUN pip install --no-cache-dir -vvv -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run server.py using flask when the container launches.
CMD ["python", "-m", "flask" ,"--app", "server.py", "run", "--host", "0.0.0.0"]
