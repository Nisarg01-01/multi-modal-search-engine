# app/main.py
from contextlib import asynccontextmanager
from io import BytesIO

from fastapi import FastAPI, File, UploadFile
from PIL import Image

from app.config import get_logger
from app.models import (
    HealthResponse,
    ImageSearchResponse,
    SearchResponse,
    TextSearchQuery,
)
from app.search import SearchService

log = get_logger(__name__)
search_service = SearchService()


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Manage startup and shutdown of the search service."""
    search_service.startup()
    log.info("Application startup complete")
    yield
    search_service.shutdown()
    log.info("Application shutdown complete")


from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="Multi-Modal Search Engine",
    version="1.0.0",
    description="Vector search API powered by CLIP and Weaviate",
    lifespan=lifespan,
)

import os

# Pick the static directory: Docker mount or local dev fallback
if os.path.exists("/app/static"):
    static_dir = "/app/static"
else:
    static_dir = os.path.join(os.getcwd(), "data", "flipkart", "images")

if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
else:
    log.warning(f"Static directory '{static_dir}' not found. Image serving will be disabled.")


@app.get("/")
def read_root():
    return {"message": "Multi-Modal Search Engine API"}


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Return service health, Weaviate connectivity, and indexed object count."""
    connected = search_service.is_healthy()
    count = search_service.get_object_count() if connected else 0
    return HealthResponse(
        status="healthy" if connected else "degraded",
        weaviate_connected=connected,
        object_count=count,
    )


@app.post("/text_search/", response_model=SearchResponse)
def text_search(search_query: TextSearchQuery):
    """Search by matching text query against product text descriptions."""
    results = search_service.text_search(search_query.query)
    return SearchResponse(query=search_query.query, results=results)


@app.post("/cross_modal_search/", response_model=SearchResponse)
def cross_modal_search(search_query: TextSearchQuery):
    """Search product images using a text query (cross-modal: text to image space)."""
    results = search_service.cross_modal_text_search(search_query.query)
    return SearchResponse(query=search_query.query, results=results)


@app.post("/hybrid_search/", response_model=SearchResponse)
def hybrid_search(search_query: TextSearchQuery):
    """Combine text-to-text and text-to-image search for the broadest coverage."""
    results = search_service.hybrid_text_search(search_query.query)
    return SearchResponse(query=search_query.query, results=results)


@app.post("/image_upload_search/", response_model=ImageSearchResponse)
async def image_upload_search(file: UploadFile = File(...)):
    """Accept an uploaded image, encode it with CLIP, and search by visual similarity."""
    try:
        image_bytes = await file.read()
        query_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        log.warning("Failed to process uploaded image: %s", file.filename)
        return ImageSearchResponse(query_image_filename=file.filename, results=[])

    results = search_service.image_search(query_image)
    return ImageSearchResponse(query_image_filename=file.filename, results=results)
