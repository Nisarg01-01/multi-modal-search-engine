# tests/test_config.py

from app.config import Settings, get_logger


def test_default_settings():
    """Settings should have sensible defaults without any env vars."""
    s = Settings()
    assert s.weaviate_host == "weaviate"
    assert s.weaviate_http_port == 8080
    assert s.weaviate_grpc_port == 50051
    assert s.clip_model_name == "clip-ViT-B-32"
    assert s.search_result_limit == 5
    assert s.log_level == "INFO"


def test_settings_from_env(monkeypatch):
    """Settings should be overridden by MMSE_-prefixed environment variables."""
    monkeypatch.setenv("MMSE_WEAVIATE_HOST", "custom-host")
    monkeypatch.setenv("MMSE_SEARCH_RESULT_LIMIT", "10")
    s = Settings()
    assert s.weaviate_host == "custom-host"
    assert s.search_result_limit == 10


def test_get_logger_returns_named_logger():
    """get_logger should return a logger with the specified name."""
    logger = get_logger("test.module")
    assert logger.name == "test.module"
    assert len(logger.handlers) > 0
