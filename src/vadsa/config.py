"""Settings from VADSA_* environment variables."""

from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

GIB = 1024**3


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VADSA_")

    environment: Literal["development", "production"] = "production"
    # NoDecode: a comma-separated value, not pydantic-settings' default JSON
    api_keys: Annotated[frozenset[str], NoDecode] = frozenset()
    engine: Literal["nemo", "fake"] = "nemo"
    # Hugging Face name or local .nemo path of the checkpoint
    model: str = "KlangAI/pianissimo-sv"
    # the name clients send as `model`
    served_model_name: str = "KlangAI/pianissimo-sv"

    max_upload_bytes: int = Field(default=GIB, gt=0)
    max_audio_seconds: float = Field(default=18_000, gt=0)
    window_seconds: float = Field(default=600, ge=1)
    max_batch_requests: int = Field(default=4, ge=1)

    max_sessions: int = Field(default=32, ge=1)
    # audio a realtime session may send
    max_session_seconds: float = Field(default=18_000, gt=0)
    # how long a session may stay open before its final commit; the audio has its own
    # limit, so this only ends a client that never finishes
    max_session_wall_seconds: float = Field(default=39_600, gt=0)
    # from reading the final commit to sending transcription.done; below the client's own
    # wait, which also spans the commit's way here and the text's way back
    finalize_seconds: float = Field(default=45, gt=0)
    idle_timeout_seconds: float = Field(default=300, gt=0)
    max_pending_seconds: float = Field(default=30, gt=0)
    # requested sizes; NeMo rounds them up to whole model frames (80 ms)
    stream_chunk_seconds: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    stream_left_padding_seconds: float = Field(default=6.0, ge=0, allow_inf_nan=False)
    stream_right_padding_seconds: float = Field(default=1.0, ge=0, allow_inf_nan=False)

    @field_validator("api_keys", mode="before")
    @classmethod
    def _split_keys(cls, value: object) -> object:
        if isinstance(value, str):
            return frozenset(key.strip() for key in value.split(",") if key.strip())
        return value

    @model_validator(mode="after")
    def _check_production(self) -> "Settings":
        if self.environment == "production":
            if not self.api_keys:
                raise ValueError("VADSA_API_KEYS is required unless VADSA_ENVIRONMENT=development")
            if self.engine == "fake":
                raise ValueError("VADSA_ENGINE=fake is only allowed in development")
        return self

    @model_validator(mode="after")
    def _check_session_limits(self) -> "Settings":
        # a live session sends its audio in real time, so a shorter backstop would end it first
        if self.max_session_wall_seconds <= self.max_session_seconds:
            raise ValueError(
                "VADSA_MAX_SESSION_WALL_SECONDS must be longer than VADSA_MAX_SESSION_SECONDS"
            )
        return self
