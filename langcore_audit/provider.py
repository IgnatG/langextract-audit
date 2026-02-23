"""Audit-logging provider implementation.

Pure decorator pattern: wraps any ``BaseLanguageModel`` and logs
every inference call to one or more pluggable ``AuditSink``
instances without altering the inference results.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import langcore as lx
from langcore.core.base_model import BaseLanguageModel
from langcore.core.types import ScoredOutput

from langcore_audit.record import AuditRecord
from langcore_audit.sinks import AuditSink, LoggingSink

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
        sample_length: When set, store the first *N* characters of
            prompt and response text in the audit record for
            debugging and correlation.  ``None`` (default) disables
            sample storage to avoid accidental PII logging.
        **kwargs: Additional keyword arguments forwarded to the
            base class.
    """

    def __init__(
        self,
        model_id: str,
        *,
        inner: BaseLanguageModel,
        sinks: list[AuditSink] | None = None,
        sample_length: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model_id = model_id
        self._inner = inner
        self._sinks: list[AuditSink] = sinks or [LoggingSink()]
        self._sample_length = sample_length

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

    @staticmethod
    def _truncate(text: str, max_length: int) -> str:
        """Truncate text to max_length, adding ellipsis if needed.

        Parameters:
            text: The text to truncate.
            max_length: Maximum character length.

        Returns:
            The truncated string.
        """
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."

    def _build_record(
        self,
        prompt: str,
        outputs: Sequence[ScoredOutput],
        latency_ms: float,
        batch_index: int,
        batch_size: int,
        *,
        success: bool = True,
        error: str | None = None,
        batch_total_ms: float | None = None,
    ) -> AuditRecord:
        """Build an ``AuditRecord`` from inference inputs and outputs.

        Parameters:
            prompt: The raw prompt text.
            outputs: The scored outputs from the inference call.
            latency_ms: Per-prompt latency in sync mode, or batch
                average (``batch_total_ms / batch_size``) in async.
            batch_index: Index of this prompt in the batch.
            batch_size: Total prompts in the batch.
            success: Whether the inference completed without error.
            error: Error message if the inference call failed.
            batch_total_ms: Total batch wall-clock time (async only).

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

        token_usage: dict[str, int] | None = None
        if best.usage is not None:
            token_usage = dict(best.usage)

        # Opt-in truncated samples for debugging/correlation
        prompt_sample: str | None = None
        response_sample: str | None = None
        if self._sample_length is not None:
            prompt_sample = self._truncate(prompt, self._sample_length)
            response_sample = self._truncate(response_text, self._sample_length)

        return AuditRecord(
            model_id=self.model_id,
            prompt_hash=AuditRecord.hash_text(prompt),
            response_hash=AuditRecord.hash_text(response_text),
            latency_ms=round(latency_ms, 2),
            timestamp=AuditRecord.utc_now_iso(),
            success=success,
            score=score,
            error=error,
            token_usage=token_usage,
            batch_index=batch_index,
            batch_size=batch_size,
            batch_total_ms=(
                round(batch_total_ms, 2) if batch_total_ms is not None else None
            ),
            prompt_sample=prompt_sample,
            response_sample=response_sample,
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
        try:
            inner_iter = iter(self._inner.infer(batch_prompts, **kwargs))
        except Exception as exc:
            # Inner provider failed on initialisation — log all
            # prompts as failed and re-raise.
            latency_ms = 0.0
            for idx, prompt in enumerate(batch_prompts):
                record = self._build_record(
                    prompt=prompt,
                    outputs=[],
                    latency_ms=latency_ms,
                    batch_index=idx,
                    batch_size=batch_size,
                    success=False,
                    error=str(exc),
                )
                self._emit(record)
            raise

        for idx, prompt in enumerate(batch_prompts):
            t0 = time.perf_counter()
            try:
                outputs_seq = next(inner_iter)
                outputs_list = list(outputs_seq)
                latency_ms = (time.perf_counter() - t0) * 1000
                record = self._build_record(
                    prompt=prompt,
                    outputs=outputs_list,
                    latency_ms=latency_ms,
                    batch_index=idx,
                    batch_size=batch_size,
                    success=True,
                )
            except StopIteration:
                # Inner provider returned fewer results than
                # prompts — emit failure records for remaining
                # prompts and stop the generator.
                latency_ms = (time.perf_counter() - t0) * 1000
                error_msg = (
                    "Inner provider returned fewer results "
                    "than batch size"
                )
                for remaining_idx in range(idx, batch_size):
                    remaining_prompt = batch_prompts[remaining_idx]
                    record = self._build_record(
                        prompt=remaining_prompt,
                        outputs=[],
                        latency_ms=latency_ms if remaining_idx == idx else 0.0,
                        batch_index=remaining_idx,
                        batch_size=batch_size,
                        success=False,
                        error=error_msg,
                    )
                    self._emit(record)
                logger.warning(
                    "Inner provider exhausted after %d of %d "
                    "prompts",
                    idx,
                    batch_size,
                )
                return
            except Exception as exc:
                latency_ms = (time.perf_counter() - t0) * 1000
                outputs_list = [ScoredOutput(score=0.0, output="")]
                record = self._build_record(
                    prompt=prompt,
                    outputs=outputs_list,
                    latency_ms=latency_ms,
                    batch_index=idx,
                    batch_size=batch_size,
                    success=False,
                    error=str(exc),
                )
                self._emit(record)
                raise
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
        try:
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
                    success=True,
                    batch_total_ms=total_latency_ms,
                )
                self._emit(record)

            return results
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            for idx, prompt in enumerate(batch_prompts):
                record = self._build_record(
                    prompt=prompt,
                    outputs=[],
                    latency_ms=latency_ms,
                    batch_index=idx,
                    batch_size=batch_size,
                    success=False,
                    error=str(exc),
                )
                self._emit(record)
            raise
