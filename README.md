# LangCore Audit Provider

A provider plugin for [LangCore](https://github.com/google/langcore) that wraps any `BaseLanguageModel` with structured audit logging. Pure decorator pattern — zero impact on inference results.

> **Note**: This is a third-party provider plugin for LangCore. For the main LangCore library, visit [google/langcore](https://github.com/google/langcore).

## Installation

Install from source:

```bash
git clone <repo-url>
cd langcore-audit
pip install -e .
```

For OpenTelemetry support:

```bash
pip install -e ".[otel]"
```

## Features

- **Pure decorator** — wraps any existing provider without modifying behaviour
- **Pluggable sinks** — log to files, Python logging, or OpenTelemetry spans
- **Structured records** — each inference call produces an `AuditRecord` with:
  - Prompt hash (SHA-256)
  - Response hash (SHA-256)
  - Latency (ms) — per-prompt in sync, averaged in async
  - `batch_total_ms` — total wall-clock time for the entire async batch
  - Token usage (if available)
  - Model ID, timestamp, success/failure, score
  - Batch index and batch size
  - Optional truncated prompt/response samples (opt-in via `sample_length`)
- **Thread-safe** — all sinks are safe for concurrent use
- **Fault-tolerant** — sink errors are logged and swallowed, never affecting inference

## Usage

### Basic Usage with Logging Sink

```python
import langcore as lx
from langcore_audit import AuditLanguageModel, LoggingSink

# Create the inner provider (any BaseLanguageModel)
inner_config = lx.factory.ModelConfig(
    model_id="litellm/azure/gpt-4o",
    provider="LiteLLMLanguageModel",
)
inner_model = lx.factory.create_model(inner_config)

# Wrap with audit logging
audit_model = AuditLanguageModel(
    model_id="audit/gpt-4o",
    inner=inner_model,
    sinks=[LoggingSink(logger_name="my_app.audit")],
)

# Use as normal — audit records are emitted automatically
result = lx.extract(
    text_or_documents="Contract text...",
    model=audit_model,
    prompt_description="Extract parties, dates, and obligations.",
)
```

### JSON File Audit Trail

```python
from langcore_audit import AuditLanguageModel, JsonFileSink

audit_model = AuditLanguageModel(
    model_id="audit/gpt-4o",
    inner=inner_model,
    sinks=[JsonFileSink("./audit_logs/extractions.jsonl")],
)
```

Each line in the output file is a JSON object:

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

```python
from langcore_audit import (
    AuditLanguageModel,
    JsonFileSink,
    LoggingSink,
)

audit_model = AuditLanguageModel(
    model_id="audit/gpt-4o",
    inner=inner_model,
    sinks=[
        LoggingSink(),                              # Console/log output
        JsonFileSink("./audit/extractions.jsonl"),  # Persistent file
    ],
)
```

### Prompt / Response Sampling

By default, prompt and response text is **not** stored in audit records (only
hashes).  Set `sample_length` to capture truncated samples for debugging:

```python
audit_model = AuditLanguageModel(
    model_id="audit/gpt-4o",
    inner=inner_model,
    sinks=[LoggingSink()],
    sample_length=200,  # store first 200 chars of prompt & response
)
```

### Async Usage

Async batches record both the per-prompt average (`latency_ms`) and the total
batch wall-clock time (`batch_total_ms`) on every record:

```python
results = await audit_model.async_infer(["prompt1", "prompt2"])
# Each AuditRecord will have:
#   latency_ms     = total / batch_size  (per-prompt average)
#   batch_total_ms = total wall-clock time for the full batch
```

## Available Sinks

| Sink | Description |
|------|-------------|
| `LoggingSink` | Emits records via Python `logging` as JSON strings |
| `JsonFileSink` | Appends newline-delimited JSON to a file |
| `OtelSpanSink` | Creates OpenTelemetry spans with audit attributes |

### Custom Sinks

Implement the `AuditSink` interface:

```python
from langcore_audit.sinks import AuditSink
from langcore_audit.record import AuditRecord

class MyCustomSink(AuditSink):
    def emit(self, record: AuditRecord) -> None:
        # Send to your preferred destination
        data = record.to_dict()
        my_api.send_audit(data)
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

Apache 2.0
