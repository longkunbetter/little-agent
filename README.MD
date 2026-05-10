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
implementations are placeholders except `stock_screener`, which uses free
BaoStock/yfinance market data and local filtering.

This project uses DeepSeek's OpenAI-compatible Chat Completions API by default:
`DEEPSEEK_BASE_URL=https://api.deepseek.com` and
`OPENAI_MODEL=deepseek-v4-pro`.

## Stock Screener

`stock_screener` supports A-share, HK, or both markets. It accepts canonical
filters such as:

```json
{
  "market": "HK",
  "criteria": {
    "filters": [
      {"field": "pe", "op": "<=", "value": 20},
      {"field": "pb", "op": "<=", "value": 2}
    ],
    "exclude": ["recent_hot_high_valuation"],
    "sort_by": "pe_asc"
  },
  "limit": 10
}
```

The tool also accepts simple aliases such as `pe_max`, `pb_max`, and
`dividend_yield_min`. Some financial indicators are not available from every
free source; unavailable fields are reported in tool warnings instead of
crashing the agent loop.

Data providers:

- A-share: BaoStock, no API key required.
- HK: yfinance, no API key required, using a curated liquid-stock universe for
  this first version.

Optional A-share runtime cap:

```bash
export STOCK_SCREENER_MAX_PROVIDER_CANDIDATES=120
```
