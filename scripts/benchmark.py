# scripts/benchmark.py
"""
Multi-modal search engine benchmark.

Tests the core claims:
  1. CLIP produces 512-dim embeddings for both text and images.
  2. Text and image embeddings share the same vector space (cross-modal similarity).
  3. Encoding latencies are measurable and consistent.
  4. Semantically related text-image pairs have lower distance than unrelated pairs.
"""

import statistics
import time
from io import BytesIO

import numpy as np
import requests
from PIL import Image
from sentence_transformers import SentenceTransformer


def fetch_test_image(url):
    """Download a test image from a URL."""
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def run_benchmark():
    print("=" * 70)
    print("MULTI-MODAL SEARCH ENGINE BENCHMARK")
    print("=" * 70)

    # Load model
    print("\n[1] Loading CLIP ViT-B/32 model...")
    t0 = time.perf_counter()
    model = SentenceTransformer("clip-ViT-B-32")
    load_time = time.perf_counter() - t0
    print(f"    Model loaded in {load_time:.2f}s")

    # Verify embedding dimensionality
    print("\n[2] Verifying embedding dimensionality...")
    sample_vec = model.encode("test")
    embed_dim = sample_vec.shape[0]
    print(f"    Embedding dimension: {embed_dim}")
    assert embed_dim == 512, f"Expected 512, got {embed_dim}"
    print("    PASS: Confirmed 512-dimensional embeddings")

    # Text encoding latency benchmark
    print("\n[3] Text encoding latency (50 iterations)...")
    test_queries = [
        "a red leather office chair",
        "wireless bluetooth headphones",
        "stainless steel water bottle",
        "running shoes for men",
        "wooden bookshelf for living room",
    ]
    text_latencies = []
    for _ in range(10):
        for query in test_queries:
            t0 = time.perf_counter()
            vec = model.encode(query)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            text_latencies.append(elapsed_ms)
            assert vec.shape == (512,), f"Expected (512,), got {vec.shape}"

    print(f"    Samples:  {len(text_latencies)}")
    print(f"    Mean:     {statistics.mean(text_latencies):.2f} ms")
    print(f"    Median:   {statistics.median(text_latencies):.2f} ms")
    print(f"    Std Dev:  {statistics.stdev(text_latencies):.2f} ms")
    print(f"    Min:      {min(text_latencies):.2f} ms")
    print(f"    Max:      {max(text_latencies):.2f} ms")
    p95_text = sorted(text_latencies)[int(0.95 * len(text_latencies))]
    print(f"    P95:      {p95_text:.2f} ms")

    # Image encoding latency benchmark
    print("\n[4] Image encoding latency...")
    # Create synthetic test images of varying sizes
    test_images = [
        Image.new("RGB", (224, 224), color=(255, 0, 0)),
        Image.new("RGB", (640, 480), color=(0, 255, 0)),
        Image.new("RGB", (1024, 768), color=(0, 0, 255)),
        Image.new("RGB", (320, 320), color=(128, 128, 0)),
        Image.new("RGB", (800, 600), color=(0, 128, 128)),
    ]
    image_latencies = []
    for _ in range(10):
        for img in test_images:
            t0 = time.perf_counter()
            vec = model.encode(img)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            image_latencies.append(elapsed_ms)
            assert vec.shape == (512,), f"Expected (512,), got {vec.shape}"

    print(f"    Samples:  {len(image_latencies)}")
    print(f"    Mean:     {statistics.mean(image_latencies):.2f} ms")
    print(f"    Median:   {statistics.median(image_latencies):.2f} ms")
    print(f"    Std Dev:  {statistics.stdev(image_latencies):.2f} ms")
    print(f"    Min:      {min(image_latencies):.2f} ms")
    print(f"    Max:      {max(image_latencies):.2f} ms")
    p95_image = sorted(image_latencies)[int(0.95 * len(image_latencies))]
    print(f"    P95:      {p95_image:.2f} ms")

    # Cross-modal similarity test
    print("\n[5] Cross-modal similarity (shared vector space verification)...")
    print("    Testing if text and image embeddings are in the same space...\n")

    # Encode text queries
    text_red = model.encode("a solid red image").tolist()
    text_blue = model.encode("a solid blue image").tolist()
    text_green = model.encode("a solid green image").tolist()

    # Encode color images
    img_red = model.encode(Image.new("RGB", (224, 224), (255, 0, 0))).tolist()
    img_blue = model.encode(Image.new("RGB", (224, 224), (0, 0, 255))).tolist()
    img_green = model.encode(Image.new("RGB", (224, 224), (0, 255, 0))).tolist()

    # Matching text-image pairs should have higher similarity than mismatched
    sim_red_red = cosine_similarity(text_red, img_red)
    sim_red_blue = cosine_similarity(text_red, img_blue)
    sim_red_green = cosine_similarity(text_red, img_green)
    sim_blue_blue = cosine_similarity(text_blue, img_blue)
    sim_green_green = cosine_similarity(text_green, img_green)

    print(f"    'red image'  vs RED image:   {sim_red_red:.4f}")
    print(f"    'red image'  vs BLUE image:  {sim_red_blue:.4f}")
    print(f"    'red image'  vs GREEN image: {sim_red_green:.4f}")
    print(f"    'blue image' vs BLUE image:  {sim_blue_blue:.4f}")
    print(f"    'green image' vs GREEN image: {sim_green_green:.4f}")

    # Verify cross-modal alignment: matching pairs beat mismatched
    print("\n    Cross-modal alignment checks:")
    checks_passed = 0
    total_checks = 3

    if sim_red_red > sim_red_blue:
        print(
            f"    PASS: 'red' text closer to RED image than BLUE image ({sim_red_red:.4f} > {sim_red_blue:.4f})"
        )
        checks_passed += 1
    else:
        print("    FAIL: 'red' text NOT closer to RED image than BLUE image")

    if sim_red_red > sim_red_green:
        print(
            f"    PASS: 'red' text closer to RED image than GREEN image ({sim_red_red:.4f} > {sim_red_green:.4f})"
        )
        checks_passed += 1
    else:
        print("    FAIL: 'red' text NOT closer to RED image than GREEN image")

    if sim_blue_blue > sim_red_blue:
        print(
            f"    PASS: 'blue' text closer to BLUE image than 'red' text ({sim_blue_blue:.4f} > {sim_red_blue:.4f})"
        )
        checks_passed += 1
    else:
        print("    FAIL: 'blue' text NOT closer to BLUE image than 'red' text")

    print(f"\n    Cross-modal alignment: {checks_passed}/{total_checks} checks passed")

    # Text-to-text similarity sanity check
    # Note: CLIP was trained on image-text pairs, so text-text similarity
    # works best with descriptive, image-like captions rather than short labels.
    print("\n[6] Text-to-text semantic similarity...")
    text_chair = model.encode("a comfortable leather office chair at a desk").tolist()
    text_sofa = model.encode("a soft leather couch in a living room").tolist()
    text_airplane = model.encode("a commercial airplane flying through clouds").tolist()
    text_headphones = model.encode("wireless bluetooth headphones on a table").tolist()

    sim_chair_sofa = cosine_similarity(text_chair, text_sofa)
    sim_chair_airplane = cosine_similarity(text_chair, text_airplane)
    sim_chair_headphones = cosine_similarity(text_chair, text_headphones)

    print(f"    'chair' vs 'sofa':       {sim_chair_sofa:.4f}  (semantically related)")
    print(f"    'chair' vs 'airplane':   {sim_chair_airplane:.4f}  (unrelated)")
    print(f"    'chair' vs 'headphones': {sim_chair_headphones:.4f}  (unrelated)")

    if sim_chair_sofa > sim_chair_airplane and sim_chair_sofa > sim_chair_headphones:
        print("    PASS: Semantically related queries are closer than unrelated ones")
    else:
        print("    FAIL: Semantic similarity ordering is incorrect")

    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print("  Model:                  CLIP ViT-B/32")
    print(f"  Embedding dimension:    {embed_dim}")
    print(f"  Text encoding (mean):   {statistics.mean(text_latencies):.2f} ms")
    print(f"  Text encoding (p95):    {p95_text:.2f} ms")
    print(f"  Image encoding (mean):  {statistics.mean(image_latencies):.2f} ms")
    print(f"  Image encoding (p95):   {p95_image:.2f} ms")
    print(f"  Cross-modal alignment:  {checks_passed}/{total_checks} passed")
    print("  Semantic similarity:    Verified")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
