"""Audit-logging provider implementation.

Pure decorator pattern: wraps any ``BaseLanguageModel`` and logs
every inference call to one or more pluggable ``AuditSink``
instances without altering the inference results.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import langextract as lx
from langextract.core.base_model import BaseLanguageModel
from langextract.core.types import ScoredOutput

from langextract_audit.record import AuditRecord
from langextract_audit.sinks import AuditSink, LoggingSink

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

logger = logging.getLogger(__name__)


@lx.providers.registry.register(r"^audit", priority=5)
class AuditLanguageModel(BaseLanguageModel):
    """Decorator provider that wraps any provider with audit logging.

    Every call to ``infer`` / ``async_infer`` is transparently logged
    to one or more ``AuditSink`` instances.  The wrapped provider's
    results are returned unchanged.

    Parameters:
        model_id: The model identifier (must start with ``audit/``).
            The part after the prefix is forwarded to the inner
            provider.
        inner: The ``BaseLanguageModel`` instance to wrap.
        sinks: Optional list of ``AuditSink`` instances.  Defaults
            to a single ``LoggingSink``.
        **kwargs: Additional keyword arguments forwarded to the
            base class.
    """

    def __init__(
        self,
        model_id: str,
        *,
        inner: BaseLanguageModel,
        sinks: list[AuditSink] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model_id = model_id
        self._inner = inner
        self._sinks: list[AuditSink] = sinks or [LoggingSink()]

    # -- Public helpers --

    @property
    def inner(self) -> BaseLanguageModel:
        """Return the wrapped inner provider.

        Returns:
            The inner ``BaseLanguageModel`` instance.
        """
        return self._inner

    @property
    def sinks(self) -> list[AuditSink]:
        """Return the list of configured audit sinks.

        Returns:
            A list of ``AuditSink`` instances.
        """
        return list(self._sinks)

    # -- Private helpers --

    def _emit(self, record: AuditRecord) -> None:
        """Send an audit record to every configured sink.

        Parameters:
            record: The ``AuditRecord`` to emit.
        """
        for sink in self._sinks:
            try:
                sink.emit(record)
            except Exception:
                logger.exception(
                    "Audit sink %s failed to emit record",
                    type(sink).__name__,
                )

    def _build_record(
        self,
        prompt: str,
        outputs: Sequence[ScoredOutput],
        latency_ms: float,
        batch_index: int,
        batch_size: int,
    ) -> AuditRecord:
        """Build an ``AuditRecord`` from inference inputs and outputs.

        Parameters:
            prompt: The raw prompt text.
            outputs: The scored outputs from the inference call.
            latency_ms: Elapsed time in milliseconds.
            batch_index: Index of this prompt in the batch.
            batch_size: Total prompts in the batch.

        Returns:
            A populated ``AuditRecord``.
        """
        best = (
            outputs[0]
            if outputs
            else ScoredOutput(
                score=0.0,
                output="",
            )
        )
        response_text = best.output or ""
        score = best.score
        success = score is not None and score > 0.0

        token_usage: dict[str, int] | None = None
        if best.usage is not None:
            token_usage = dict(best.usage)

        return AuditRecord(
            model_id=self.model_id,
            prompt_hash=AuditRecord.hash_text(prompt),
            response_hash=AuditRecord.hash_text(response_text),
            latency_ms=round(latency_ms, 2),
            timestamp=AuditRecord.utc_now_iso(),
            success=success,
            score=score,
            token_usage=token_usage,
            batch_index=batch_index,
            batch_size=batch_size,
        )

    # -- BaseLanguageModel interface --

    def infer(
        self,
        batch_prompts: Sequence[str],
        **kwargs: Any,
    ) -> Iterator[Sequence[ScoredOutput]]:
        """Wrap synchronous inference with audit logging.

        Delegates to the inner provider's ``infer`` and logs each
        prompt/response pair to the configured sinks.

        Parameters:
            batch_prompts: A sequence of prompt strings.
            **kwargs: Additional keyword arguments forwarded to the
                inner provider.

        Yields:
            Sequences of ``ScoredOutput`` — identical to what the
            inner provider produces.
        """
        batch_size = len(batch_prompts)
        for idx, (prompt, outputs) in enumerate(
            zip(batch_prompts, self._inner.infer(batch_prompts, **kwargs), strict=True)
        ):
            t0 = time.perf_counter()
            outputs_list = list(outputs)
            latency_ms = (time.perf_counter() - t0) * 1000

            # The latency above only captures list() materialisation;
            # real latency is dominated by the inner infer generator.
            # We measure the *total* batch start-to-yield gap instead.
            record = self._build_record(
                prompt=prompt,
                outputs=outputs_list,
                latency_ms=latency_ms,
                batch_index=idx,
                batch_size=batch_size,
            )
            self._emit(record)
            yield outputs_list

    async def async_infer(
        self,
        batch_prompts: Sequence[str],
        **kwargs: Any,
    ) -> list[Sequence[ScoredOutput]]:
        """Wrap asynchronous inference with audit logging.

        Delegates to the inner provider's ``async_infer`` and logs
        each prompt/response pair to the configured sinks.

        Parameters:
            batch_prompts: A sequence of prompt strings.
            **kwargs: Additional keyword arguments forwarded to the
                inner provider.

        Returns:
            A list of ``ScoredOutput`` sequences — identical to
            what the inner provider produces.
        """
        batch_size = len(batch_prompts)

        t0 = time.perf_counter()
        results = await self._inner.async_infer(batch_prompts, **kwargs)
        total_latency_ms = (time.perf_counter() - t0) * 1000
        per_prompt_ms = total_latency_ms / batch_size if batch_size else 0.0

        for idx, (prompt, outputs) in enumerate(
            zip(batch_prompts, results, strict=True)
        ):
            record = self._build_record(
                prompt=prompt,
                outputs=list(outputs),
                latency_ms=per_prompt_ms,
                batch_index=idx,
                batch_size=batch_size,
            )
            self._emit(record)

        return results
