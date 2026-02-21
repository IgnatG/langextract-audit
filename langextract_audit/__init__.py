"""LangExtract audit-logging provider plugin.

Wraps any ``BaseLanguageModel`` with structured audit logging.
Supports pluggable sinks: JSON file, structured logging, and
OpenTelemetry spans.
"""

from langextract_audit.provider import AuditLanguageModel
from langextract_audit.sinks import (
    AuditSink,
    JsonFileSink,
    LoggingSink,
)

__all__ = [
    "AuditLanguageModel",
    "AuditSink",
    "JsonFileSink",
    "LoggingSink",
]
__version__ = "1.1.1"
