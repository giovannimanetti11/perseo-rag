from perseo_rag.config import Settings


def test_settings_use_safe_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.environment == "development"
    assert settings.log_level == "INFO"
    assert settings.expose_api_docs is False
    assert settings.database_url != settings.migration_database_url


def test_settings_read_prefixed_environment(monkeypatch) -> None:
    monkeypatch.setenv("PERSEO_ENVIRONMENT", "test")
    monkeypatch.setenv("PERSEO_EXPOSE_API_DOCS", "true")

    settings = Settings(_env_file=None)

    assert settings.environment == "test"
    assert settings.expose_api_docs is True
