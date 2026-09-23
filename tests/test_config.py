import pytest
from pydantic import ValidationError

from vadsa.config import Settings


def test_keys_are_comma_separated() -> None:
    settings = Settings(api_keys=" first, second ,,")
    assert settings.api_keys == {"first", "second"}


def test_production_refuses_to_start_without_keys() -> None:
    with pytest.raises(ValidationError, match="VADSA_API_KEYS"):
        Settings(environment="production")


def test_production_refuses_the_fake_engine() -> None:
    with pytest.raises(ValidationError, match="fake"):
        Settings(environment="production", api_keys="key", engine="fake")


def test_development_may_run_without_keys() -> None:
    assert Settings(environment="development", engine="fake").api_keys == frozenset()
