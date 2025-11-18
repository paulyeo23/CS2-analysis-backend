FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY app/requirements.txt .
RUN pip install -r requirements.txt

# Copy code
COPY app .

ENTRYPOINT ["python", "main.py"]