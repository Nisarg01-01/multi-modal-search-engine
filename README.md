# Multi-Modal AI Search Engine

A search engine that goes beyond keywords. Describe what you want in plain language -- or upload a photo -- and it finds matching products by understanding the actual content, not just text patterns.

Built on **CLIP** (contrastive language-image pre-training) and **Weaviate** (vector database with HNSW indexing).

## What It Does

| Search Mode | How It Works |
|---|---|
| **Text Search** | Encode your query into a vector, then find the nearest product descriptions in embedding space. |
| **Image Search** | Upload a photo. The engine encodes it the same way and retrieves visually similar products. |
| **Hybrid Search** | Runs both text-to-text and text-to-image queries, merges the results, and returns the best matches. |

The system currently indexes the **Flipkart Products Dataset** (~18,000 items across electronics, clothing, furniture, and more).

---

## Quick Start

### Prerequisites
*   Docker Desktop (installed and running)
*   Git

### 1. Start the services
```bash
git clone https://github.com/yourusername/multi-modal-search-engine.git
cd multi-modal-search-engine
docker compose up --build
```

### 2. Load the dataset

The database starts empty. Download the [Flipkart E-Commerce Dataset](https://www.kaggle.com/datasets/PromptCloudHQ/flipkart-products) from Kaggle and place the CSV here:

```
data/flipkart/flipkart_com-ecommerce_sample.csv
```

Then run the ingestion script. It downloads product images in parallel and indexes everything into Weaviate:

```bash
conda activate multimodal-search
python scripts/ingest_flipkart.py
```

This is a one-time step. Data persists in a Docker volume across restarts.

### 3. Use it
*   **UI**: [http://localhost:8501](http://localhost:8501)
*   **API**: [http://localhost:8000](http://localhost:8000)

---

## Architecture

```
Browser --> Streamlit (8501) --> FastAPI (8000) --> Weaviate (8080)
                                     |
                               CLIP ViT-B/32
                          (text + image encoder)
```

| Component | Role |
|---|---|
| **FastAPI** | Serves search endpoints, encodes queries with CLIP, queries Weaviate. |
| **Weaviate** | Stores product vectors and metadata. Handles approximate nearest-neighbor search via HNSW. |
| **Streamlit** | Frontend with text input, image upload, and a product grid for results. |
| **CLIP ViT-B/32** | Encodes both text and images into a shared 512-dimensional vector space. |

All services are containerized with Docker Compose.

---

## Project Structure

```
app/
  main.py        -- FastAPI application, routes, static file serving
  search.py      -- CLIP encoding, Weaviate queries, result formatting
  models.py      -- Pydantic request/response schemas
  config.py      -- Environment-based settings (ports, model, limits)
scripts/
  ingest_flipkart.py  -- Dataset ingestion (parallel downloads, batch indexing)
ui.py            -- Streamlit frontend
docker-compose.yml
Dockerfile
requirements.txt
```

---

## License
MIT