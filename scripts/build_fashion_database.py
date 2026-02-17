import csv
import time
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
import weaviate
import weaviate.classes.config as wvc
import weaviate.classes.init as wvi
from PIL import Image

# Removed argparse and tqdm to isolate crash
# Hardcoded settings for debugging
HOST = "localhost"
MODEL_NAME = "clip-ViT-B-32"
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data" / "fashion-dataset"
IMAGES_DIR = DATA_DIR / "images"
STYLES_CSV = DATA_DIR / "styles.csv"

def build_database():
    print("Step 1: Start", flush=True)
    
    # Load Model FIRST
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Step 2: Loading model on {device}...", flush=True)
    model = SentenceTransformer(MODEL_NAME, device=device)
    print("Step 3: Model loaded", flush=True)

    # Connect
    print(f"Step 4: Connecting to {HOST}...", flush=True)
    client = weaviate.connect_to_local(
        host=HOST,
        port=8080,
        grpc_port=50051,
        additional_config=wvi.AdditionalConfig(
            timeout=wvi.Timeout(init=30, query=60, insert=120)
        ),
    )
    
    if not client.is_ready():
        print("Weaviate not ready", flush=True)
        return
    print("Step 5: Connected", flush=True)

    # Schema
    if client.collections.exists("Product"):
        client.collections.delete("Product")
    client.collections.create(
        name="Product",
        properties=[
            wvc.Property(name="name", data_type=wvc.DataType.TEXT),
            wvc.Property(name="description", data_type=wvc.DataType.TEXT),
            wvc.Property(name="category", data_type=wvc.DataType.TEXT),
            wvc.Property(name="image", data_type=wvc.DataType.TEXT),
            wvc.Property(name="year", data_type=wvc.DataType.INT),
        ],
        vectorizer_config=[
            wvc.Configure.NamedVectors.none(name="text_vector"),
            wvc.Configure.NamedVectors.none(name="image_vector"),
        ],
    )
    print("Step 6: Schema created", flush=True)
    products_collection = client.collections.get("Product")

    # Read CSV
    print("Step 7: Reading CSV...", flush=True)
    products = []
    with open(STYLES_CSV, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            products.append(row)
    print(f"Step 8: Found {len(products)} products", flush=True)

    # Ingest loop
    ingested = 0
    with products_collection.batch.dynamic() as batch:
        for i, row in enumerate(products):
            # if i >= 100: break # Removed limit
            
            try:
                image_id = str(row.get("id", "")).strip()
                image_path = IMAGES_DIR / f"{image_id}.jpg"
                if not image_path.exists():
                    image_path = IMAGES_DIR / f"{image_id}.png"
                    if not image_path.exists():
                        continue
                
                # Encode
                image = Image.open(image_path).convert("RGB")
                image_vector = model.encode(image).tolist()
                
                name = row.get("productDisplayName") or "Unknown"
                desc = f"{name} {row.get('articleType')}"
                text_vector = model.encode(desc).tolist()
                
                batch.add_object(
                    properties={
                        "name": name,
                        "description": desc,
                        "image": f"{image_id}.jpg", 
                    },
                    vector={
                        "text_vector": text_vector,
                        "image_vector": image_vector,
                    },
                )
                ingested += 1
                if ingested % 1000 == 0:
                    print(f"Ingested {ingested}", flush=True)
            except Exception as e:
                print(f"Error: {e}", flush=True)

    print(f"Finished. Ingested {ingested}", flush=True)
    client.close()

if __name__ == "__main__":
    build_database()
