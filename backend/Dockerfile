FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libpq-dev gcc git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download embedding model into the image layer
RUN python - <<'EOF'
from sentence_transformers import SentenceTransformer
SentenceTransformer('all-mpnet-base-v2')
print("Embedding model cached.")
EOF

COPY . .
RUN mkdir -p data

EXPOSE 8080

CMD ["python", "consumer.py"]
