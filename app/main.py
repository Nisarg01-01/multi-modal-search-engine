# app/main.py
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import weaviate
from sentence_transformers import SentenceTransformer
from PIL import Image
from io import BytesIO

# --- Application Setup ---
IMAGE_BASE_URL = "https://m.media-amazon.com/images/I/"
model = SentenceTransformer("clip-ViT-B-32")
client = None
app = FastAPI(title="Multi-Modal Search Engine")


# --- Pydantic Models ---
class TextSearchQuery(BaseModel):
    query: str


# --- Application Lifecycle ---
@app.on_event("startup")
def startup_event():
    global client
    client = weaviate.connect_to_local(host="weaviate", port=8080, grpc_port=50051)
    print("FastAPI app has started and connected to Weaviate.")


@app.on_event("shutdown")
def shutdown_event():
    if client:
        client.close()
    print("Weaviate connection closed.")


# --- Helper Function for Formatting Results ---
def format_results(response):
    results = []
    for item in response.objects:
        results.append(
            {
                "name": item.properties["name"],
                "image_id": item.properties["image"],
                "distance": item.metadata.distance,
            }
        )
    return results


# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Welcome!"}


@app.post("/text_search/")
def text_search(search_query: TextSearchQuery):
    """Performs a vector search based on a text query."""
    query_vector = model.encode(search_query.query).tolist()
    products = client.collections.get("Product")
    response = products.query.near_vector(
        near_vector=query_vector,
        limit=5,
        return_metadata=["distance"],
        target_vector="text_vector",
        return_properties=["name", "description", "image"],
    )
    return {"query": search_query.query, "results": format_results(response)}


@app.post("/image_upload_search/")
async def image_upload_search(file: UploadFile = File(...)):
    """Receives an uploaded image, generates an embedding, and performs a vector search."""
    try:
        # Read the uploaded file into memory
        image_bytes = await file.read()
        query_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return {"error": "Could not read or process uploaded image."}

    query_vector = model.encode(query_image).tolist()
    products = client.collections.get("Product")

    response = products.query.near_vector(
        near_vector=query_vector,
        limit=5,
        return_metadata=["distance"],
        target_vector="image_vector",
        return_properties=["name", "description", "image"],
    )
    return {"query_image_filename": file.filename, "results": format_results(response)}
