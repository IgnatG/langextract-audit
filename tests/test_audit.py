"""Tests for AuditLanguageModel provider and audit sinks."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

import pytest
from langextract.core.base_model import BaseLanguageModel
from langextract.core.types import ScoredOutput

from langextract_audit import AuditLanguageModel
from langextract_audit.record import AuditRecord
from langextract_audit.sinks import AuditSink, JsonFileSink, LoggingSink

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubProvider(BaseLanguageModel):
    """Minimal stub provider for testing the audit wrapper."""

    def __init__(
        self,
        responses: list[list[ScoredOutput]] | None = None,
    ) -> None:
        super().__init__()
        self._responses = responses or [
            [ScoredOutput(score=1.0, output="stub response")]
        ]
        self.call_count = 0

    def infer(
        self,
        batch_prompts: Sequence[str],
        **kwargs,
    ) -> Iterator[Sequence[ScoredOutput]]:
        for i, _prompt in enumerate(batch_prompts):
            self.call_count += 1
            idx = min(i, len(self._responses) - 1)
            yield self._responses[idx]

    async def async_infer(
        self,
        batch_prompts: Sequence[str],
        **kwargs,
    ) -> list[Sequence[ScoredOutput]]:
        results: list[Sequence[ScoredOutput]] = []
        for i, _prompt in enumerate(batch_prompts):
            self.call_count += 1
            idx = min(i, len(self._responses) - 1)
            results.append(self._responses[idx])
        return results


class _CaptureSink(AuditSink):
    """Audit sink that captures records in-memory for assertions."""

    def __init__(self) -> None:
        self.records: list[AuditRecord] = []

    def emit(self, record: AuditRecord) -> None:
        self.records.append(record)


# ---------------------------------------------------------------------------
# AuditRecord tests
# ---------------------------------------------------------------------------


class TestAuditRecord:
    """Tests for the AuditRecord data structure."""

    def test_hash_text_deterministic(self) -> None:
        h1 = AuditRecord.hash_text("hello")
        h2 = AuditRecord.hash_text("hello")
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex length

    def test_hash_text_different_inputs(self) -> None:
        h1 = AuditRecord.hash_text("hello")
        h2 = AuditRecord.hash_text("world")
        assert h1 != h2

    def test_to_dict_round_trip(self) -> None:
        record = AuditRecord(
            model_id="test-model",
            prompt_hash="abc",
            response_hash="def",
            latency_ms=42.5,
            timestamp="2026-01-01T00:00:00+00:00",
            success=True,
            score=1.0,
            token_usage={"prompt_tokens": 10, "completion_tokens": 5},
            batch_index=0,
            batch_size=1,
        )
        d = record.to_dict()
        assert d["model_id"] == "test-model"
        assert d["latency_ms"] == 42.5
        assert d["token_usage"]["prompt_tokens"] == 10

    def test_utc_now_iso_format(self) -> None:
        ts = AuditRecord.utc_now_iso()
        assert "T" in ts
        assert "+" in ts or "Z" in ts


# ---------------------------------------------------------------------------
# Sink tests
# ---------------------------------------------------------------------------


class TestLoggingSink:
    """Tests for the LoggingSink."""

    def test_emit_calls_logger(self) -> None:
        record = AuditRecord(
            model_id="m",
            prompt_hash="ph",
            response_hash="rh",
            latency_ms=1.0,
            timestamp="t",
            success=True,
        )
        sink = LoggingSink(logger_name="test.audit")
        with mock.patch.object(sink._logger, "log") as m:
            sink.emit(record)
        m.assert_called_once()
        logged_json = m.call_args[0][1]
        parsed = json.loads(logged_json)
        assert parsed["model_id"] == "m"


class TestJsonFileSink:
    """Tests for the JsonFileSink."""

    def test_writes_newline_delimited_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "audit.jsonl"
            sink = JsonFileSink(path)
            try:
                for i in range(3):
                    record = AuditRecord(
                        model_id=f"model-{i}",
                        prompt_hash="ph",
                        response_hash="rh",
                        latency_ms=float(i),
                        timestamp="t",
                        success=True,
                    )
                    sink.emit(record)
            finally:
                sink.close()

            lines = path.read_text().strip().split("\n")
            assert len(lines) == 3
            for i, line in enumerate(lines):
                parsed = json.loads(line)
                assert parsed["model_id"] == f"model-{i}"

    def test_creates_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sub" / "dir" / "audit.jsonl"
            sink = JsonFileSink(path)
            try:
                record = AuditRecord(
                    model_id="m",
                    prompt_hash="ph",
                    response_hash="rh",
                    latency_ms=0.0,
                    timestamp="t",
                    success=True,
                )
                sink.emit(record)
            finally:
                sink.close()
            assert path.exists()


# ---------------------------------------------------------------------------
# Provider tests — sync
# ---------------------------------------------------------------------------


class TestAuditProviderSync:
    """Tests for AuditLanguageModel sync inference."""

    def test_delegates_to_inner_provider(self) -> None:
        stub = _StubProvider()
        capture = _CaptureSink()
        audit = AuditLanguageModel(
            model_id="audit/test",
            inner=stub,
            sinks=[capture],
        )
        results = list(audit.infer(["prompt1", "prompt2"]))
        assert len(results) == 2
        assert stub.call_count == 2
        assert results[0][0].output == "stub response"

    def test_records_emitted_per_prompt(self) -> None:
        stub = _StubProvider()
        capture = _CaptureSink()
        audit = AuditLanguageModel(
            model_id="audit/test",
            inner=stub,
            sinks=[capture],
        )
        list(audit.infer(["p1", "p2", "p3"]))
        assert len(capture.records) == 3
        for i, rec in enumerate(capture.records):
            assert rec.batch_index == i
            assert rec.batch_size == 3

    def test_record_contains_prompt_hash(self) -> None:
        stub = _StubProvider()
        capture = _CaptureSink()
        audit = AuditLanguageModel(
            model_id="audit/test",
            inner=stub,
            sinks=[capture],
        )
        list(audit.infer(["hello world"]))
        expected_hash = AuditRecord.hash_text("hello world")
        assert capture.records[0].prompt_hash == expected_hash

    def test_record_marks_success_on_zero_score(self) -> None:
        """A score of 0 is still a successful inference (no error)."""
        stub = _StubProvider(responses=[[ScoredOutput(score=0.0, output="result")]])
        capture = _CaptureSink()
        audit = AuditLanguageModel(
            model_id="audit/test",
            inner=stub,
            sinks=[capture],
        )
        list(audit.infer(["prompt"]))
        assert capture.records[0].success is True
        assert capture.records[0].score == 0.0

    def test_record_captures_token_usage(self) -> None:
        """Token usage is captured when ScoredOutput supports it."""
        usage = {"prompt_tokens": 10, "completion_tokens": 5}
        so = ScoredOutput(score=1.0, output="ok", usage=usage)
        stub = _StubProvider(responses=[[so]])
        capture = _CaptureSink()
        audit = AuditLanguageModel(
            model_id="audit/test",
            inner=stub,
            sinks=[capture],
        )
        list(audit.infer(["prompt"]))
        assert capture.records[0].token_usage == usage

    def test_kwargs_forwarded_to_inner(self) -> None:
        stub = _StubProvider()
        capture = _CaptureSink()
        audit = AuditLanguageModel(
            model_id="audit/test",
            inner=stub,
            sinks=[capture],
        )
        with mock.patch.object(stub, "infer", wraps=stub.infer) as m:
            list(audit.infer(["p"], pass_num=2))
        m.assert_called_once_with(["p"], pass_num=2)

    def test_sink_error_does_not_propagate(self) -> None:
        stub = _StubProvider()

        class _FailSink(AuditSink):
            def emit(self, record: AuditRecord) -> None:
                raise RuntimeError("sink exploded")

        audit = AuditLanguageModel(
            model_id="audit/test",
            inner=stub,
            sinks=[_FailSink()],
        )
        # Should not raise
        results = list(audit.infer(["prompt"]))
        assert len(results) == 1

    def test_multiple_sinks_all_receive_records(self) -> None:
        stub = _StubProvider()
        c1, c2 = _CaptureSink(), _CaptureSink()
        audit = AuditLanguageModel(
            model_id="audit/test",
            inner=stub,
            sinks=[c1, c2],
        )
        list(audit.infer(["prompt"]))
        assert len(c1.records) == 1
        assert len(c2.records) == 1

    def test_latency_measures_inner_provider(self) -> None:
        """Latency should capture inner provider time, not just list()."""
        import time as _time

        class _SlowProvider(BaseLanguageModel):
            def infer(self, batch_prompts, **kw):
                for _p in batch_prompts:
                    _time.sleep(0.05)
                    yield [ScoredOutput(score=1.0, output="ok")]

            async def async_infer(self, batch_prompts, **kw):
                return [[ScoredOutput(score=1.0, output="ok")]]

        capture = _CaptureSink()
        audit = AuditLanguageModel(
            model_id="audit/test",
            inner=_SlowProvider(),
            sinks=[capture],
        )
        list(audit.infer(["p1"]))
        # Should capture >= 50ms, not ~0ms
        assert capture.records[0].latency_ms >= 40.0

    def test_error_records_failure(self) -> None:
        """Inner provider exception sets success=False and error."""

        class _FailProvider(BaseLanguageModel):
            def infer(self, batch_prompts, **kw):
                raise RuntimeError("model exploded")

            async def async_infer(self, batch_prompts, **kw):
                raise RuntimeError("model exploded")

        capture = _CaptureSink()
        audit = AuditLanguageModel(
            model_id="audit/test",
            inner=_FailProvider(),
            sinks=[capture],
        )
        with pytest.raises(RuntimeError, match="model exploded"):
            list(audit.infer(["prompt"]))
        assert len(capture.records) == 1
        assert capture.records[0].success is False
        assert capture.records[0].error == "model exploded"

    def test_sample_length_stores_truncated_text(self) -> None:
        stub = _StubProvider(
            responses=[[ScoredOutput(score=1.0, output="response text here")]]
        )
        capture = _CaptureSink()
        audit = AuditLanguageModel(
            model_id="audit/test",
            inner=stub,
            sinks=[capture],
            sample_length=5,
        )
        list(audit.infer(["hello world"]))
        assert capture.records[0].prompt_sample == "hello..."
        assert capture.records[0].response_sample == "respo..."

    def test_no_sample_by_default(self) -> None:
        stub = _StubProvider()
        capture = _CaptureSink()
        audit = AuditLanguageModel(
            model_id="audit/test",
            inner=stub,
            sinks=[capture],
        )
        list(audit.infer(["hello"]))
        assert capture.records[0].prompt_sample is None
        assert capture.records[0].response_sample is None

    def test_default_sink_is_logging(self) -> None:
        stub = _StubProvider()
        audit = AuditLanguageModel(
            model_id="audit/test",
            inner=stub,
        )
        assert len(audit.sinks) == 1
        assert isinstance(audit.sinks[0], LoggingSink)

    def test_inner_exhausted_early_emits_failures(self) -> None:
        """When the inner provider returns fewer results than prompts,
        failure records must be emitted for the remaining prompts and
        the generator stops cleanly without raising."""

        class _ShortProvider(BaseLanguageModel):
            """Returns only one result regardless of batch size."""

            def infer(self, batch_prompts, **kw):
                yield [ScoredOutput(score=1.0, output="only one")]
                # StopIteration on second next()

            async def async_infer(self, batch_prompts, **kw):
                return [[ScoredOutput(score=1.0, output="only one")]]

        capture = _CaptureSink()
        audit = AuditLanguageModel(
            model_id="audit/test",
            inner=_ShortProvider(),
            sinks=[capture],
        )
        results = list(audit.infer(["p1", "p2", "p3"]))
        # Only the first prompt should yield a result
        assert len(results) == 1
        assert results[0][0].output == "only one"
        # All 3 prompts should have records: 1 success + 2 failures
        assert len(capture.records) == 3
        assert capture.records[0].success is True
        assert capture.records[1].success is False
        assert capture.records[2].success is False
        assert "fewer results" in (capture.records[1].error or "")
        assert capture.records[1].batch_index == 1
        assert capture.records[2].batch_index == 2


# ---------------------------------------------------------------------------
# Provider tests — async
# ---------------------------------------------------------------------------


class TestAuditProviderAsync:
    """Tests for AuditLanguageModel async inference."""

    @pytest.mark.asyncio
    async def test_async_delegates_to_inner(self) -> None:
        stub = _StubProvider()
        capture = _CaptureSink()
        audit = AuditLanguageModel(
            model_id="audit/test",
            inner=stub,
            sinks=[capture],
        )
        results = await audit.async_infer(["p1", "p2"])
        assert len(results) == 2
        assert stub.call_count == 2

    @pytest.mark.asyncio
    async def test_async_records_emitted(self) -> None:
        stub = _StubProvider()
        capture = _CaptureSink()
        audit = AuditLanguageModel(
            model_id="audit/test",
            inner=stub,
            sinks=[capture],
        )
        await audit.async_infer(["p1", "p2", "p3"])
        assert len(capture.records) == 3
        for i, rec in enumerate(capture.records):
            assert rec.batch_index == i
            assert rec.batch_size == 3

    @pytest.mark.asyncio
    async def test_async_returns_inner_results(self) -> None:
        stub = _StubProvider(
            responses=[[ScoredOutput(score=0.9, output="async result")]]
        )
        capture = _CaptureSink()
        audit = AuditLanguageModel(
            model_id="audit/test",
            inner=stub,
            sinks=[capture],
        )
        results = await audit.async_infer(["prompt"])
        assert results[0][0].output == "async result"

    @pytest.mark.asyncio
    async def test_async_batch_total_ms_populated(self) -> None:
        """batch_total_ms should be set on every async record.

        For a batch of N prompts ``batch_total_ms`` must be >= 0 and
        should be >= ``latency_ms`` (the per-prompt average) since
        the total cannot be less than one prompt's share.
        """
        stub = _StubProvider()
        capture = _CaptureSink()
        audit = AuditLanguageModel(
            model_id="audit/test",
            inner=stub,
            sinks=[capture],
        )
        await audit.async_infer(["a", "b", "c"])
        assert len(capture.records) == 3
        for rec in capture.records:
            assert rec.batch_total_ms is not None
            assert rec.batch_total_ms >= 0.0
            # Per-prompt average must not exceed the total.
            assert rec.latency_ms <= rec.batch_total_ms + 1e-9

    @pytest.mark.asyncio
    async def test_async_single_prompt_total_equals_latency(self) -> None:
        """For a single-prompt async batch, latency_ms == batch_total_ms."""
        stub = _StubProvider()
        capture = _CaptureSink()
        audit = AuditLanguageModel(
            model_id="audit/test",
            inner=stub,
            sinks=[capture],
        )
        await audit.async_infer(["only prompt"])
        rec = capture.records[0]
        assert rec.batch_total_ms is not None
        assert rec.latency_ms == pytest.approx(rec.batch_total_ms, abs=1e-6)


# ---------------------------------------------------------------------------
# Plugin registration test
# ---------------------------------------------------------------------------


class TestPluginRegistration:
    """Tests for entry-point discovery."""

    def test_audit_prefix_resolves(self) -> None:
        import langextract as lx
        from langextract.providers import registry

        lx.providers.load_plugins_once()
        cls = registry.resolve("audit/my-model")
        assert cls.__name__ == "AuditLanguageModel"
