# app/models.py
from pydantic import BaseModel, Field


class TextSearchQuery(BaseModel):
    """Payload for the text-based search endpoint."""

    query: str = Field(..., min_length=1, max_length=500)


class SearchResult(BaseModel):
    """A single search result returned by the vector search."""

    name: str
    image_id: str
    distance: float
    description: str | None = None
    brand: str | None = None
    price: float | None = None


class SearchResponse(BaseModel):
    """Wrapper for a list of search results."""

    query: str
    results: list[SearchResult]


class ImageSearchResponse(BaseModel):
    """Response for image-based search queries."""

    query_image_filename: str
    results: list[SearchResult]


class HealthResponse(BaseModel):
    """Response from the health check endpoint."""

    status: str
    weaviate_connected: bool
    object_count: int
