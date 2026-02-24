# app/config.py
import logging

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    weaviate_host: str = "localhost"
    weaviate_http_port: int = 8080
    weaviate_grpc_port: int = 50051
    clip_model_name: str = "clip-ViT-B-32"
    clip_device: str = "cpu"
    search_result_limit: int = 5
    image_base_url: str = "http://localhost:8000/static/images/"
    log_level: str = "INFO"

    model_config = {"env_prefix": "MMSE_"}


settings = Settings()


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(settings.log_level)
    return logger
