# tests/test_models.py
import pytest
from pydantic import ValidationError

from app.models import HealthResponse, SearchResponse, SearchResult, TextSearchQuery


def test_text_search_query_valid():
    q = TextSearchQuery(query="red sneakers")
    assert q.query == "red sneakers"


def test_text_search_query_empty_string_rejected():
    with pytest.raises(ValidationError):
        TextSearchQuery(query="")


def test_text_search_query_too_long_rejected():
    with pytest.raises(ValidationError):
        TextSearchQuery(query="x" * 501)


def test_search_result_serialization():
    r = SearchResult(name="Product A", image_id="abc123", distance=0.15)
    data = r.model_dump()
    assert data == {"name": "Product A", "image_id": "abc123", "distance": 0.15}


def test_search_response_contains_results():
    results = [SearchResult(name="A", image_id="id1", distance=0.1)]
    resp = SearchResponse(query="test", results=results)
    assert len(resp.results) == 1
    assert resp.query == "test"


def test_health_response():
    h = HealthResponse(status="healthy", weaviate_connected=True, object_count=500)
    assert h.status == "healthy"
    assert h.object_count == 500
