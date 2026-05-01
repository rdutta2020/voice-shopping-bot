# Voice Shopping Bot

[![CI](https://github.com/rdutta2020/voice-shopping-bot/actions/workflows/ci.yml/badge.svg)](https://github.com/rdutta2020/voice-shopping-bot/actions/workflows/ci.yml)

A voice-first shopping assistant for retail shop owners in India. Built over 7 days.

## Stack

| Layer | Technology |
|---|---|
| Android client | Kotlin · SpeechRecognizer |
| Backend | FastAPI · Python 3.12 |
| AI pipeline | Anthropic Claude API · LangGraph |
| Observability | Langfuse |
| CI/CD | GitHub Actions |

## CI Pipeline

| Job | Trigger | Blocks merge? |
|---|---|---|
| **Lint** (flake8) | push + PR | ✅ yes |
| **Security** (Snyk) | push + PR | ⚠️ warning only |
| **Accuracy Eval** | push + PR | ✅ yes — fails if intent accuracy < 80 % |
| **Claude Code Review** | PR only | ❌ informational |

### Required secrets

Set these in **Settings → Secrets → Actions**:

| Secret | Used by |
|---|---|
| `ANTHROPIC_API_KEY` | Eval · Code review |
| `SNYK_TOKEN` | Snyk security scan |
| `LANGFUSE_PUBLIC_KEY` | Langfuse tracing (optional) |
| `LANGFUSE_SECRET_KEY` | Langfuse tracing (optional) |
| `LANGFUSE_HOST` | Langfuse tracing (optional) |

## Running locally

```bash
cd backend
ANTHROPIC_API_KEY=sk-ant-... venv/bin/uvicorn main:app --reload
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/detect-intent` | Classify utterance intent |
| POST | `/extract-items` | Parse items from speech |
| POST | `/chat` | Multi-turn shopping assistant |
| POST | `/process` | Full LangGraph pipeline |

### Run the eval suite

```bash
cd backend
ANTHROPIC_API_KEY=sk-ant-... python3 eval.py
# Writes eval_report.json with per-case scores and accuracy summary
```
