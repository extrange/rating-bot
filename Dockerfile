FROM mcr.microsoft.com/devcontainers/python:1-3.11

# Install requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt
