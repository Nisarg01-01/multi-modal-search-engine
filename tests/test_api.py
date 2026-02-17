# tests/test_api.py
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.models import SearchResult


@pytest.fixture()
def client():
    """Create a TestClient with the SearchService mocked out."""
    with patch("app.main.search_service") as mock_service:
        mock_service.startup = MagicMock()
        mock_service.shutdown = MagicMock()
        mock_service.is_healthy.return_value = True
        mock_service.get_object_count.return_value = 1000
        mock_service.text_search.return_value = [
            SearchResult(name="Test Product", image_id="abc123", distance=0.10),
        ]
        mock_service.cross_modal_text_search.return_value = [
            SearchResult(name="Visual Match via Text", image_id="cross01", distance=0.12),
        ]
        mock_service.hybrid_text_search.return_value = [
            SearchResult(name="Hybrid Result", image_id="hyb01", distance=0.08),
        ]
        mock_service.image_search.return_value = [
            SearchResult(name="Visual Match", image_id="xyz789", distance=0.05),
        ]

        from app.main import app

        with TestClient(app) as tc:
            yield tc


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Multi-Modal Search Engine" in response.json()["message"]


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["weaviate_connected"] is True
    assert data["object_count"] == 1000


def test_text_search_endpoint(client):
    response = client.post("/text_search/", json={"query": "red sneakers"})
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "red sneakers"
    assert len(data["results"]) == 1
    assert data["results"][0]["name"] == "Test Product"


def test_cross_modal_search_endpoint(client):
    response = client.post("/cross_modal_search/", json={"query": "red chair with armrest"})
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "red chair with armrest"
    assert len(data["results"]) == 1
    assert data["results"][0]["name"] == "Visual Match via Text"


def test_hybrid_search_endpoint(client):
    response = client.post("/hybrid_search/", json={"query": "flower pattern dress"})
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "flower pattern dress"
    assert len(data["results"]) == 1
    assert data["results"][0]["name"] == "Hybrid Result"


def test_text_search_empty_query_rejected(client):
    response = client.post("/text_search/", json={"query": ""})
    assert response.status_code == 422


def test_image_upload_search_endpoint(client):
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/image_upload_search/",
        files={"file": ("test.jpg", buf, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["query_image_filename"] == "test.jpg"
    assert len(data["results"]) == 1
