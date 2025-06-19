# HealthPay Claim Agent Backend

A minimal, modular backend for processing medical insurance claim documents using AI-powered multi-agent orchestration. Built with FastAPI, this project demonstrates LLM-based document classification, extraction, validation, and automated claim decisioning.

## 🚀 Features

- **`/process-claim` endpoint:** Upload and process multiple claim PDFs in a single request
- **LLM-based agents:** Document classification, extraction, and validation
- **Modular, async FastAPI architecture**
- **Structured JSON output:** Includes validation and claim decision

## 🛠️ Getting Started

### Prerequisites

- Python 3.10+
- Google API Key (for Gemini/ADK)
- Mistral API Key (for Mistral OCR)
- (Optional) Docker

### Installation

```bash
uv sync --frozen --no-dev
```

### Running the Server

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```