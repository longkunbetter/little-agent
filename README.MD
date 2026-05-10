# little-agent

A lightweight personal stock-picking agent framework for China A-share and Hong
Kong markets.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Configure DeepSeek credentials in the local `.env` file:

```bash
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_API_KEY=your-deepseek-api-key
OPENAI_MODEL=deepseek-v4-pro
```

Import the config before running:

```bash
set -a
source .env
set +a
```

## Run

```bash
.venv/bin/python -m little_agent
```

The current version wires the agent loop, DeepSeek's OpenAI-compatible Chat
Completions interaction, and built-in tool contracts. Market data
implementations are placeholders.

This project uses DeepSeek's OpenAI-compatible Chat Completions API by default:
`DEEPSEEK_BASE_URL=https://api.deepseek.com` and
`OPENAI_MODEL=deepseek-v4-pro`.
