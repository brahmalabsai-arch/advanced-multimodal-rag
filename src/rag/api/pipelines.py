"""One pipeline per provider, sharing one index (F2 bring-your-own-key).

A `Pipeline` is cheap except for its `IndexStore` — the Chroma vectors, chunks, sentences and
the embedder, which is the whole memory budget on a small instance. The registry therefore
loads the store once and hands the same object to every pipeline it builds.

Why one pipeline per provider rather than one pipeline with a per-request provider: the cache's
version keys are computed once per pipeline (§5.7) and `generator_model` now carries the
provider, so a pipeline *is* the unit that keeps one provider's cached answers separate from
another's. Threading a per-request `VersionKeys` through `CacheService`, `L2Cache`, the sweeper
and the stats endpoint would put the same invariant in six places where a missed default would
silently serve a Groq answer to an Anthropic visitor. Building three small objects avoids that
class of bug entirely; the model client is still per request.

Pipelines are built on first use, so a provider nobody asks for costs nothing.
"""

from __future__ import annotations

import threading

from rag.core.config import ModelsConfig, load_models_config
from rag.core.logging import get_logger
from rag.core.settings import Settings, get_settings
from rag.graph import Pipeline
from rag.llm import PROVIDERS, profile_for_provider
from rag.query.store import IndexStore, get_store

log = get_logger(__name__)


class PipelineRegistry:
    """Lazily builds one `Pipeline` per provider over a shared `IndexStore`."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        models: ModelsConfig | None = None,
        store: IndexStore | None = None,
    ):
        self.settings = settings or get_settings()
        self.models = models or load_models_config(settings=self.settings)
        self.store = store or get_store()
        self._pipelines: dict[str, Pipeline] = {}
        self._lock = threading.Lock()

    @property
    def default_provider(self) -> str:
        """The provider of the configured `MODEL_PROFILE`, used when no key is supplied."""
        for provider, profile in ((p, profile_for_provider(p, self.models)) for p in PROVIDERS):
            if profile == self.models.active_profile:
                return provider
        return next(iter(PROVIDERS))

    def for_provider(self, provider: str) -> Pipeline:
        """The pipeline that serves `provider`, building it on first use."""
        profile = profile_for_provider(provider, self.models)
        with self._lock:
            pipeline = self._pipelines.get(provider)
            if pipeline is None:
                pipeline = Pipeline(
                    store=self.store,
                    settings=self.settings.model_copy(update={"model_profile": profile}),
                    models=self.models.model_copy(update={"active_profile": profile}),
                    # no client is forced: locally the profile's key in `.env` builds one,
                    # and where there is no key every request must bring its own (BYOK)
                    client=None,
                )
                self._pipelines[provider] = pipeline
                log.info(
                    "pipeline ready for provider=%s profile=%s (context budget %d tokens)",
                    provider,
                    profile,
                    pipeline.context_budget,
                )
            return pipeline

    def default(self) -> Pipeline:
        return self.for_provider(self.default_provider)

    def loaded(self) -> list[str]:
        return sorted(self._pipelines)


__all__ = ["PipelineRegistry"]
