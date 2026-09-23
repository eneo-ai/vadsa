"""The app factory: `uvicorn vadsa.main:create_app --factory`."""

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from importlib.metadata import version

from fastapi import FastAPI

from vadsa.api import health, models, realtime, transcriptions
from vadsa.audio import SAMPLE_RATE
from vadsa.config import Settings
from vadsa.engine.base import Engine
from vadsa.engine.scheduler import Scheduler
from vadsa.errors import install_error_handlers


def create_app(
    settings: Settings | None = None, load_engine: Callable[[], Engine] | None = None
) -> FastAPI:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = settings or Settings()

    def load() -> Engine:
        if load_engine is not None:
            return load_engine()
        if settings.engine == "fake":
            from vadsa.engine.fake import FakeEngine

            return FakeEngine()
        from vadsa.engine.nemo import NemoEngine

        return NemoEngine(settings)

    # the model loads on the scheduler's thread while the server already answers /health
    scheduler = Scheduler(
        load,
        max_sessions=settings.max_sessions,
        max_pending_samples=int(settings.max_pending_seconds * SAMPLE_RATE),
        max_batch_requests=settings.max_batch_requests,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        scheduler.start()
        yield
        scheduler.stop()

    app = FastAPI(
        title="vadsa",
        description="Swedish speech recognition with KlangAI/pianissimo-sv, served like vLLM.",
        version=version("vadsa"),
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.scheduler = scheduler
    install_error_handlers(app)
    for module in (health, models, transcriptions, realtime):
        app.include_router(module.router)
    return app
