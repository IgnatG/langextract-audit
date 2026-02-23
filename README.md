# LangCore Audit

> Provider plugin for [LangCore](https://github.com/ignatg/langcore) — structured audit logging for every LLM call with zero impact on inference results.

[![PyPI version](https://img.shields.io/pypi/v/langcore-audit)](https://pypi.org/project/langcore-audit/)
[![Python](https://img.shields.io/pypi/pyversions/langcore-audit)](https://pypi.org/project/langcore-audit/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

---

## Overview

**langcore-audit** is a provider plugin for [LangCore](https://github.com/ignatg/langcore) that wraps any `BaseLanguageModel` with structured audit logging. Every inference call produces a detailed `AuditRecord` — without modifying the extraction results in any way. Plug in one or more sinks to route audit data to logs, files, or OpenTelemetry.

---

## Features

- **Pure decorator pattern** — wraps any existing LangCore provider without altering inference behaviour or results
- **Pluggable sink architecture** — route audit records to Python logging, JSONL files, OpenTelemetry spans, or custom destinations
- **Structured audit records** — each call captures:
  - SHA-256 prompt and response hashes
  - Latency (per-prompt in sync, averaged in async)
  - Batch wall-clock time (`batch_total_ms`)
  - Token usage (prompt, completion, total)
  - Model ID, timestamp, success/failure status, and score
  - Batch index and batch size
  - Optional truncated prompt/response samples
- **Thread-safe** — all built-in sinks are safe for concurrent use
- **Fault-tolerant** — sink errors are logged and swallowed, never affecting inference
- **Multiple simultaneous sinks** — emit records to several destinations at once
- **Opt-in prompt/response sampling** — capture truncated text samples for debugging without storing full payloads
- **Zero-config plugin** — auto-registered via Python entry points

---

## Installation

```bash
pip install langcore-audit
```

For OpenTelemetry support:

```bash
pip install "langcore-audit[otel]"
```

---

## Quick Start

### Integration with LangCore

langcore-audit integrates with LangCore through the **decorator provider pattern**. Wrap any existing LangCore model to add audit logging:

```python
import langcore as lx
from langcore_audit import AuditLanguageModel, LoggingSink

# 1. Create the base LLM provider
inner_config = lx.factory.ModelConfig(
    model_id="litellm/gpt-4o",
    provider="LiteLLMLanguageModel",
)
inner_model = lx.factory.create_model(inner_config)

# 2. Wrap with audit logging
audit_model = AuditLanguageModel(
    model_id="audit/gpt-4o",
    inner=inner_model,
    sinks=[LoggingSink(logger_name="extraction.audit")],
)

# 3. Use as a drop-in replacement — audit records are emitted automatically
result = lx.extract(
    text_or_documents="Acme Corp agrees to pay $50,000 to Beta LLC by March 2025.",
    model=audit_model,
    prompt_description="Extract parties, amounts, and dates.",
    examples=[
        lx.data.ExampleData(
            text="Alpha Inc will pay $10,000 to Omega Ltd by January 2024.",
            extractions=[
                lx.data.Extraction("party", "Alpha Inc", attributes={"role": "payer"}),
                lx.data.Extraction("party", "Omega Ltd", attributes={"role": "payee"}),
                lx.data.Extraction("monetary_amount", "$10,000"),
                lx.data.Extraction("date", "January 2024"),
            ],
        )
    ],
)
```

Every extraction call automatically emits an `AuditRecord` to all configured sinks.

---

## Usage

### JSON File Audit Trail

Persist audit records as newline-delimited JSON for compliance or analysis:

```python
from langcore_audit import AuditLanguageModel, JsonFileSink

audit_model = AuditLanguageModel(
    model_id="audit/gpt-4o",
    inner=inner_model,
    sinks=[JsonFileSink("./audit_logs/extractions.jsonl")],
)
```

Each line in the output file is a self-contained JSON object:

```json
{
  "model_id": "audit/gpt-4o",
  "prompt_hash": "a3f2...",
  "response_hash": "b7c1...",
  "latency_ms": 1234.56,
  "timestamp": "2026-02-21T10:30:00+00:00",
  "success": true,
  "score": 1.0,
  "token_usage": {"prompt_tokens": 150, "completion_tokens": 45, "total_tokens": 195},
  "batch_index": 0,
  "batch_size": 1,
  "batch_total_ms": null,
  "prompt_sample": null,
  "response_sample": null
}
```

### OpenTelemetry Integration

Emit audit data as OpenTelemetry spans for integration with Jaeger, Datadog, Grafana Tempo, or any OTel-compatible backend:

```python
from langcore_audit import AuditLanguageModel
from langcore_audit.sinks import OtelSpanSink

audit_model = AuditLanguageModel(
    model_id="audit/gpt-4o",
    inner=inner_model,
    sinks=[OtelSpanSink(tracer_name="my_service.llm")],
)
```

### Multiple Sinks

Route audit records to several destinations simultaneously:

```python
from langcore_audit import AuditLanguageModel, JsonFileSink, LoggingSink

audit_model = AuditLanguageModel(
    model_id="audit/gpt-4o",
    inner=inner_model,
    sinks=[
        LoggingSink(),                              # Console / log output
        JsonFileSink("./audit/extractions.jsonl"),  # Persistent JSONL file
    ],
)
```

### Prompt / Response Sampling

By default, only hashes are stored — no raw text. Enable truncated sampling for debugging:

```python
audit_model = AuditLanguageModel(
    model_id="audit/gpt-4o",
    inner=inner_model,
    sinks=[LoggingSink()],
    sample_length=200,  # Store first 200 chars of prompt & response
)
```

### Async Batch Logging

Async batches record both per-prompt averages and total batch timing:

```python
results = await audit_model.async_infer(["prompt1", "prompt2"])
# Each AuditRecord contains:
#   latency_ms     = total / batch_size  (per-prompt average)
#   batch_total_ms = total wall-clock time for the full batch
```

---

## Available Sinks

| Sink | Description |
|------|-------------|
| `LoggingSink` | Emits records via Python `logging` as JSON strings |
| `JsonFileSink` | Appends newline-delimited JSON to a file (thread-safe) |
| `OtelSpanSink` | Creates OpenTelemetry spans with audit attributes (requires `[otel]` extra) |

### Creating Custom Sinks

Implement the `AuditSink` interface to send records anywhere:

```python
from langcore_audit.sinks import AuditSink
from langcore_audit.record import AuditRecord

class DatabaseSink(AuditSink):
    def __init__(self, connection_string: str) -> None:
        self._db = connect(connection_string)

    def emit(self, record: AuditRecord) -> None:
        self._db.insert("audit_records", record.to_dict())
```

---

## Composing with Other Plugins

langcore-audit is designed to be the outermost wrapper in a provider stack, capturing the full lifecycle of each call including retries from guardrails:

```python
import langcore as lx
from langcore_audit import AuditLanguageModel, JsonFileSink
from langcore_guardrails import GuardrailLanguageModel, SchemaValidator, OnFailAction

# Base provider
llm = lx.factory.create_model(
    lx.factory.ModelConfig(model_id="litellm/gpt-4o", provider="LiteLLMLanguageModel")
)

# Layer 1: Output validation with retry
guarded = GuardrailLanguageModel(
    model_id="guardrails/gpt-4o",
    inner=llm,
    validators=[SchemaValidator(MySchema, on_fail=OnFailAction.REASK)],
    max_retries=3,
)

# Layer 2: Audit logging (outermost — captures retries too)
audited = AuditLanguageModel(
    model_id="audit/gpt-4o",
    inner=guarded,
    sinks=[JsonFileSink("./audit.jsonl")],
    sample_length=500,
)

# Use the full stack
result = lx.extract(
    text_or_documents="Contract text...",
    model=audited,
    prompt_description="Extract entities.",
    examples=[...],
)
```

---

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Requirements

- Python ≥ 3.12
- `langcore`
- Optional: `opentelemetry-api`, `opentelemetry-sdk` (for `[otel]` extra)

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
