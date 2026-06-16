# M04

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v "$($PWD.Path)/qdrant_storage:/qdrant/storage" qdrant/qdrant:latest

# Lancer PostgreSQL avec pgvector
docker run -d --name pgvector -p 5432:5432 -e POSTGRES_PASSWORD=postgres pgvector/pgvector:pg17
```
