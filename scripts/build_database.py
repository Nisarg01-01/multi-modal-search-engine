# scripts/build_database.py
import weaviate
import weaviate.classes.config as wvc
from sentence_transformers import SentenceTransformer
from weaviate.util import get_valid_uuid
from uuid import uuid4
from tqdm import tqdm
import requests
from PIL import Image
from io import BytesIO
import gzip
from pathlib import Path
import re
import json

# --- Configuration ---
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
EXTRACTED_DATA_PATH = DATA_DIR / "listings" / "listings" / "metadata"
IMAGE_BASE_URL = "https://m.media-amazon.com/images/I/"
MODEL_NAME = "clip-ViT-B-32"

# --- Load Model ---
print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)
print("Model loaded.")


# --- Helper Functions ---
def parse_multilingual_field(field_data):
    if not isinstance(field_data, list) or not field_data:
        return None
    for item in field_data:
        if item.get("language_tag") == "en_US":
            return item.get("value")
    return None


def is_valid_product(name, description, image_id):
    if not name or not description or not image_id or image_id == "No Image":
        return False
    if "placeholder" in description.lower() or "not available" in name.lower():
        return False
    if re.fullmatch(r"[A-Z0-9-]{8,}", name):
        return False
    if len(description) < 20:
        return False
    return True


def create_schema(client):
    """Creates the multi-vector schema."""
    if client.collections.exists("Product"):
        client.collections.delete("Product")
    client.collections.create(
        name="Product",
        properties=[
            wvc.Property(name="name", data_type=wvc.DataType.TEXT),
            wvc.Property(name="description", data_type=wvc.DataType.TEXT),
            wvc.Property(name="image", data_type=wvc.DataType.TEXT),
        ],
        vectorizer_config=[
            wvc.Configure.NamedVectors.none(name="text_vector"),
            wvc.Configure.NamedVectors.none(name="image_vector"),
        ],
    )
    print("Successfully created the multi-vector 'Product' collection.")


# --- Main Atomic Pipeline ---
def final_pipeline():
    client = None
    try:
        client = weaviate.connect_to_local(host="weaviate", port=8080, grpc_port=50051)
        create_schema(client)
        products_collection = client.collections.get("Product")

        json_gz_files = list(EXTRACTED_DATA_PATH.glob("*.json.gz"))

        print("Processing ~147,000 items to find and load valid products...")
        with products_collection.batch.dynamic() as batch:
            # Create a session for efficient network requests
            session = requests.Session()
            # Iterate through all files and lines
            for gz_file in tqdm(json_gz_files, desc="Processing source files"):
                with gzip.open(gz_file, "rt", encoding="utf-8") as f:
                    for line in f:
                        product = json.loads(line)
                        name = parse_multilingual_field(product.get("item_name"))
                        description = parse_multilingual_field(
                            product.get("bullet_point", "")
                        )
                        image_id = product.get("main_image_id")

                        if not is_valid_product(name, description, image_id):
                            continue

                        try:
                            # Attempt to download and embed the image
                            image_url = f"{IMAGE_BASE_URL}{image_id}.jpg"
                            http_response = session.get(image_url, timeout=10)
                            http_response.raise_for_status()
                            image = Image.open(BytesIO(http_response.content)).convert(
                                "RGB"
                            )
                            image_vector = model.encode(image).tolist()

                            # Embed text
                            text_for_embedding = f"{name}: {description}"
                            text_vector = model.encode(text_for_embedding).tolist()

                            # Add to batch only if BOTH are successful
                            batch.add_object(
                                properties={
                                    "name": name,
                                    "description": description,
                                    "image": image_id,
                                },
                                uuid=get_valid_uuid(uuid4()),
                                vector={
                                    "text_vector": text_vector,
                                    "image_vector": image_vector,
                                },
                            )
                        except Exception:
                            # If the image download or processing fails, skip this item
                            continue

        final_count = products_collection.aggregate.over_all(
            total_count=True
        ).total_count
        print(
            f"\nFinished. Successfully loaded {final_count} fully multi-modal products into Weaviate."
        )

    except Exception as e:
        print(f"An error occurred during the main process: {e}")
    finally:
        if client:
            client.close()
            print("Connection to Weaviate closed.")


if __name__ == "__main__":
    final_pipeline()
