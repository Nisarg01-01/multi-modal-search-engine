# app/search.py
import weaviate
from PIL import Image
from sentence_transformers import SentenceTransformer

from app.config import get_logger, settings
from app.models import SearchResult

log = get_logger(__name__)


class SearchService:
    """Encapsulates CLIP embedding generation and Weaviate vector queries."""

    def __init__(self) -> None:
        self.model: SentenceTransformer | None = None
        self.client: weaviate.WeaviateClient | None = None

    def startup(self) -> None:
        """Load the CLIP model and connect to Weaviate."""
        log.info("Loading CLIP model: %s on %s", settings.clip_model_name, settings.clip_device)
        self.model = SentenceTransformer(settings.clip_model_name, device=settings.clip_device)
        embed_dim = self.model.get_sentence_embedding_dimension() or "N/A"
        log.info("CLIP model loaded (embedding dim=%s)", embed_dim)

        log.info(
            "Connecting to Weaviate at %s:%d",
            settings.weaviate_host,
            settings.weaviate_http_port,
        )
        self.client = weaviate.connect_to_local(
            host=settings.weaviate_host,
            port=settings.weaviate_http_port,
            grpc_port=settings.weaviate_grpc_port,
        )
        log.info("Weaviate connection established")

    def shutdown(self) -> None:
        """Close the Weaviate connection."""
        if self.client:
            self.client.close()
            log.info("Weaviate connection closed")

    def encode_text(self, text: str) -> list[float]:
        """Generate a CLIP embedding from a text string."""
        return self.model.encode(text).tolist()

    def encode_image(self, image: Image.Image) -> list[float]:
        """Generate a CLIP embedding from a PIL image."""
        return self.model.encode(image).tolist()

    def _format_results(self, response) -> list[SearchResult]:
        """Convert a Weaviate response into a list of SearchResult objects."""
        results = []
        for item in response.objects:
            results.append(
                SearchResult(
                    name=item.properties["name"],
                    image_id=item.properties["image"],
                    distance=item.metadata.distance,
                )
            )
        return results

    def text_search(self, query: str) -> list[SearchResult]:
        """Run a vector search against the text_vector named vector."""
        query_vector = self.encode_text(query)
        products = self.client.collections.get("Product")
        response = products.query.near_vector(
            near_vector=query_vector,
            limit=settings.search_result_limit,
            return_metadata=["distance"],
            target_vector="text_vector",
            return_properties=["name", "description", "image"],
        )
        return self._format_results(response)

    def cross_modal_text_search(self, query: str) -> list[SearchResult]:
        """Search product images using a text query (cross-modal: text -> image space)."""
        query_vector = self.encode_text(query)
        products = self.client.collections.get("Product")
        response = products.query.near_vector(
            near_vector=query_vector,
            limit=settings.search_result_limit,
            return_metadata=["distance"],
            target_vector="image_vector",
            return_properties=["name", "description", "image"],
        )
        return self._format_results(response)

    def hybrid_text_search(self, query: str) -> list[SearchResult]:
        """Combine text-to-text and text-to-image results, keeping the best match per product."""
        text_results = self.text_search(query)
        cross_results = self.cross_modal_text_search(query)

        seen: dict[str, SearchResult] = {}
        for result in text_results + cross_results:
            key = result.image_id
            if key not in seen or result.distance < seen[key].distance:
                seen[key] = result

        merged = sorted(seen.values(), key=lambda r: r.distance)
        return merged[: settings.search_result_limit]

    def image_search(self, image: Image.Image) -> list[SearchResult]:
        """Run a vector search against the image_vector named vector."""
        query_vector = self.encode_image(image)
        products = self.client.collections.get("Product")
        response = products.query.near_vector(
            near_vector=query_vector,
            limit=settings.search_result_limit,
            return_metadata=["distance"],
            target_vector="image_vector",
            return_properties=["name", "description", "image"],
        )
        return self._format_results(response)

    def get_object_count(self) -> int:
        """Return the total number of objects in the Product collection."""
        try:
            products = self.client.collections.get("Product")
            result = products.aggregate.over_all(total_count=True)
            return result.total_count
        except Exception:
            return -1

    def is_healthy(self) -> bool:
        """Check if the Weaviate connection is alive."""
        try:
            return self.client.is_ready()
        except Exception:
            return False
