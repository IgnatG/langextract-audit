"""Pluggable audit sinks for the audit-logging provider.

Each sink receives an ``AuditRecord`` and persists or forwards it
according to its own semantics.  Multiple sinks can be composed
via the provider's ``sinks`` list.
"""

from __future__ import annotations

import abc
import json
import logging
import threading
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langcore_audit.record import AuditRecord

__all__ = [
    "AuditSink",
    "JsonFileSink",
    "LoggingSink",
    "OtelSpanSink",
]

logger = logging.getLogger(__name__)


class AuditSink(abc.ABC):
    """Abstract base class for audit sinks."""

    @abc.abstractmethod
    def emit(self, record: AuditRecord) -> None:
        """Persist or forward a single audit record.

        Parameters:
            record: The audit record to emit.
        """


class LoggingSink(AuditSink):
    """Emit audit records via Python's standard ``logging`` module.

    Parameters:
        logger_name: Name for the logger instance.
            Defaults to ``"langcore.audit"``.
        level: Logging level for audit entries.
            Defaults to ``logging.INFO``.
    """

    def __init__(
        self,
        logger_name: str = "langcore.audit",
        level: int = logging.INFO,
    ) -> None:
        self._logger = logging.getLogger(logger_name)
        self._level = level

    def emit(self, record: AuditRecord) -> None:
        """Log the audit record as a structured JSON string.

        Parameters:
            record: The audit record to emit.
        """
        self._logger.log(
            self._level,
            json.dumps(record.to_dict(), default=str),
        )


class JsonFileSink(AuditSink):
    """Append audit records as newline-delimited JSON to a file.

    Thread-safe.  The file is opened on first ``emit`` and remains
    open until ``close()`` is called.

    Parameters:
        path: Filesystem path for the audit log file.
            Parent directories are created automatically.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._file: IO[str] | None = None

    def _ensure_open(self) -> IO[str]:
        """Open the audit log file, creating parent dirs if needed.

        Returns:
            The open file handle.
        """
        if self._file is None or self._file.closed:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self._path.open("a", encoding="utf-8")
        return self._file

    def emit(self, record: AuditRecord) -> None:
        """Append the audit record as a JSON line.

        Parameters:
            record: The audit record to emit.
        """
        line = json.dumps(record.to_dict(), default=str)
        with self._lock:
            fh = self._ensure_open()
            fh.write(line + "\n")
            fh.flush()

    def close(self) -> None:
        """Close the underlying file handle."""
        with self._lock:
            if self._file is not None and not self._file.closed:
                self._file.close()


class OtelSpanSink(AuditSink):
    """Emit audit records as OpenTelemetry span events.

    Requires the ``opentelemetry-api`` and ``opentelemetry-sdk``
    packages (install via ``pip install langcore-audit[otel]``).

    Parameters:
        tracer_name: Name for the OpenTelemetry tracer.
            Defaults to ``"langcore.audit"``.
    """

    def __init__(self, tracer_name: str = "langcore.audit") -> None:
        try:
            from opentelemetry import trace

            self._tracer = trace.get_tracer(tracer_name)
        except ImportError as exc:
            raise ImportError(
                "OpenTelemetry packages are required for OtelSpanSink. "
                "Install with: pip install langcore-audit[otel]"
            ) from exc

    def emit(self, record: AuditRecord) -> None:
        """Create an OTel span carrying the audit record attributes.

        Parameters:
            record: The audit record to emit.
        """
        attrs: dict[str, Any] = {
            "audit.model_id": record.model_id,
            "audit.prompt_hash": record.prompt_hash,
            "audit.response_hash": record.response_hash,
            "audit.latency_ms": record.latency_ms,
            "audit.success": record.success,
        }
        if record.token_usage is not None:
            for k, v in record.token_usage.items():
                attrs[f"audit.tokens.{k}"] = v

        with self._tracer.start_as_current_span(
            "langcore.audit.infer",
            attributes=attrs,
        ):
            pass
