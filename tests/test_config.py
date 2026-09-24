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


def test_the_wall_clock_backstop_outlasts_the_audio_limit() -> None:
    # a live session sends its audio in real time, so a shorter backstop would end it first
    with pytest.raises(ValidationError, match="VADSA_MAX_SESSION_WALL_SECONDS"):
        Settings(
            environment="development",
            engine="fake",
            max_session_seconds=3600,
            max_session_wall_seconds=3600,
        )


def test_development_may_run_without_keys() -> None:
    assert Settings(environment="development", engine="fake").api_keys == frozenset()


@pytest.mark.parametrize(
    "field", ["stream_chunk_seconds", "stream_left_padding_seconds", "stream_right_padding_seconds"]
)
def test_stream_timing_settings_must_be_finite(field: str) -> None:
    with pytest.raises(ValidationError, match=field):
        Settings(environment="development", **{field: float("inf")})
