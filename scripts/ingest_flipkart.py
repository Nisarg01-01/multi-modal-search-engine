import csv
import ast
import time
from pathlib import Path
import requests
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
import weaviate
import weaviate.classes.config as wvc
import weaviate.classes.init as wvi
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
HOST = "localhost"
MODEL_NAME = "clip-ViT-B-32"
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data" / "flipkart"
IMAGES_DIR = DATA_DIR / "images"
CSV_PATH = DATA_DIR / "flipkart_com-ecommerce_sample.csv"
MAX_WORKERS = 20  # Number of parallel download threads

def setup_weaviate(client):
    """Refreshes the Product collection with Flipkart-specific schema."""
    if client.collections.exists("Product"):
        print("Deleting existing Product collection...", flush=True)
        client.collections.delete("Product")
    
    print("Creating new Product collection...", flush=True)
    client.collections.create(
        name="Product",
        properties=[
            wvc.Property(name="name", data_type=wvc.DataType.TEXT),
            wvc.Property(name="description", data_type=wvc.DataType.TEXT),
            wvc.Property(name="category", data_type=wvc.DataType.TEXT), 
            wvc.Property(name="brand", data_type=wvc.DataType.TEXT),
            wvc.Property(name="price", data_type=wvc.DataType.NUMBER),
            wvc.Property(name="image", data_type=wvc.DataType.TEXT),
        ],
        vectorizer_config=[
            wvc.Configure.NamedVectors.none(name="text_vector"),
            wvc.Configure.NamedVectors.none(name="image_vector"),
        ],
    )

def download_image(args):
    """Downloads image from URL to save_path. Returns (uniq_id, success)."""
    url, save_path, uniq_id = args
    if save_path.exists():
        return uniq_id, True # Already exists

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            return uniq_id, True
    except Exception:
        pass 
    return uniq_id, False

def build_database():
    # 0. Setup
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Read CSV & Prep Downloads
    print("Reading CSV to prepare downloads...", flush=True)
    rows = []
    download_tasks = []
    
    with open(CSV_PATH, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            
            # Extract Image URL
            try:
                image_urls = ast.literal_eval(row.get("image", "[]"))
                if not image_urls:
                    continue
                image_url = image_urls[0]
                uniq_id = row.get("uniq_id")
                image_filename = f"{uniq_id}.jpg"
                save_path = IMAGES_DIR / image_filename
                
                download_tasks.append((image_url, save_path, uniq_id))
            except:
                continue

    print(f"Found {len(rows)} products. Preparing {len(download_tasks)} image downloads...", flush=True)
    if not rows:
        print("No rows found in CSV. Check the file path.")
        return
    
    # 2. Parallel Downloads
    successful_downloads = set()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        futures = [executor.submit(download_image, task) for task in download_tasks]
        
        count = 0
        total = len(futures)
        for future in as_completed(futures):
            uniq_id, success = future.result()
            if success:
                successful_downloads.add(uniq_id)
            
            count += 1
            if count % 100 == 0:
                print(f"Downloaded/Checked {count}/{total} images...", flush=True)

    print(f"Downloads complete. {len(successful_downloads)} images available for ingestion.", flush=True)

    # 3. Load Model & Connect Weaviate
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...", flush=True)
    model = SentenceTransformer(MODEL_NAME, device=device)

    print(f"Connecting to Weaviate at {HOST}...", flush=True)
    client = weaviate.connect_to_local(
        host=HOST,
        port=8080,
        grpc_port=50051,
        additional_config=wvi.AdditionalConfig(
            timeout=wvi.Timeout(init=30, query=60, insert=120)
        ),
    )
    
    setup_weaviate(client)
    products_collection = client.collections.get("Product")

    # 4. Ingest Loop (Sequential Encode -> Batch Upload)
    print("Starting ingestion...", flush=True)
    ingested_count = 0
    
    with products_collection.batch.dynamic() as batch:
        for i, row in enumerate(rows):
            uniq_id = row.get("uniq_id")
            
            # Skip if image download failed
            if uniq_id not in successful_downloads:
                continue

            image_filename = f"{uniq_id}.jpg"
            local_image_path = IMAGES_DIR / image_filename
            
            # Encode
            try:
                # Image Vector
                try:
                    pil_image = Image.open(local_image_path).convert("RGB")
                    image_vector = model.encode(pil_image).tolist()
                except Exception:
                    # Corrupt image file
                    continue
                
                # Text Data
                name = row.get("product_name") or "Unknown"
                brand = row.get("brand") or "Generic"
                category_tree = row.get("product_category_tree", "")
                try:
                    category = ast.literal_eval(category_tree)[0].split(" >> ")[0]
                except:
                    category = "General"

                desc = row.get("description") or ""
                full_text = f"{name} {brand} {category} {desc}"
                text_vector = model.encode(full_text).tolist()

                # Price
                try:
                    price = float(row.get("discounted_price", 0))
                except:
                    price = 0.0

                # Add to Batch
                batch.add_object(
                    properties={
                        "name": name,
                        "description": desc,
                        "category": category,
                        "brand": brand,
                        "price": price,
                        "image": image_filename,
                    },
                    vector={
                        "text_vector": text_vector,
                        "image_vector": image_vector,
                    },
                )
                ingested_count += 1
                
                if ingested_count % 100 == 0:
                    print(f"Ingested {ingested_count} items...", flush=True)

            except Exception as e:
                print(f"Error processing {uniq_id}: {e}", flush=True)

    print(f"Finished! Total ingested: {ingested_count}")
    client.close()

if __name__ == "__main__":
    build_database()
