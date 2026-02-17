# tests/test_search.py
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.models import SearchResult
from app.search import SearchService


def _make_weaviate_response(items):
    """Build a mock Weaviate query response."""
    objects = []
    for name, image, distance in items:
        obj = SimpleNamespace(
            properties={"name": name, "image": image, "description": "desc"},
            metadata=SimpleNamespace(distance=distance),
        )
        objects.append(obj)
    return SimpleNamespace(objects=objects)


class TestSearchService:
    def setup_method(self):
        self.service = SearchService()
        self.service.model = MagicMock()
        self.service.client = MagicMock()

    def test_format_results_returns_search_result_list(self):
        response = _make_weaviate_response(
            [
                ("Shoe", "img1", 0.12),
                ("Hat", "img2", 0.34),
            ]
        )
        results = self.service._format_results(response)
        assert len(results) == 2
        assert isinstance(results[0], SearchResult)
        assert results[0].name == "Shoe"
        assert results[1].distance == 0.34

    def test_format_results_empty(self):
        response = _make_weaviate_response([])
        results = self.service._format_results(response)
        assert results == []

    def test_encode_text_calls_model(self):
        import numpy as np

        self.service.model.encode.return_value = np.array([0.1, 0.2, 0.3])
        vec = self.service.encode_text("test query")
        self.service.model.encode.assert_called_once_with("test query")
        assert vec == [0.1, 0.2, 0.3]

    def test_text_search_queries_weaviate(self):
        import numpy as np

        self.service.model.encode.return_value = np.array([0.1, 0.2])
        mock_collection = MagicMock()
        self.service.client.collections.get.return_value = mock_collection
        mock_collection.query.near_vector.return_value = _make_weaviate_response(
            [
                ("Item", "img", 0.05),
            ]
        )

        results = self.service.text_search("office chair")
        assert len(results) == 1
        assert results[0].name == "Item"
        mock_collection.query.near_vector.assert_called_once()

    def test_is_healthy_returns_false_on_exception(self):
        self.service.client.is_ready.side_effect = Exception("connection lost")
        assert self.service.is_healthy() is False

    def test_get_object_count_returns_negative_on_error(self):
        self.service.client.collections.get.side_effect = Exception("fail")
        assert self.service.get_object_count() == -1
