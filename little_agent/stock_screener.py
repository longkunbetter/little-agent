from __future__ import annotations

import math
import os
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import date, timedelta
from io import StringIO
from typing import Any, Protocol

try:
    import pandas as pd
except ImportError:  # pragma: no cover - handled at runtime by provider
    pd = None  # type: ignore[assignment]


SUPPORTED_FIELDS = {
    "pe",
    "pe_ttm",
    "pb",
    "dividend_yield",
    "roe",
    "market_cap",
    "turnover",
    "price_change_60d",
    "ytd_change",
}

FIELD_ALIASES = {
    "市盈率": "pe",
    "动态市盈率": "pe",
    "pe_ratio": "pe",
    "low_pe": "pe",
    "低市盈率": "pe",
    "市盈率ttm": "pe_ttm",
    "pe_ttm": "pe_ttm",
    "市净率": "pb",
    "low_pb": "pb",
    "低市净率": "pb",
    "pb_ratio": "pb",
    "股息率": "dividend_yield",
    "股息": "dividend_yield",
    "高股息": "dividend_yield",
    "dividend": "dividend_yield",
    "dividend_yield": "dividend_yield",
    "roe": "roe",
    "净资产收益率": "roe",
    "市值": "market_cap",
    "market_cap": "market_cap",
    "成交额": "turnover",
    "turnover": "turnover",
    "60日涨跌幅": "price_change_60d",
    "price_change_60d": "price_change_60d",
    "年初至今涨跌幅": "ytd_change",
    "ytd_change": "ytd_change",
}

SORT_ALIASES = {
    "dividend_yield_desc": ("dividend_yield", False),
    "pe_asc": ("pe", True),
    "pe_ttm_asc": ("pe_ttm", True),
    "pb_asc": ("pb", True),
    "roe_desc": ("roe", False),
    "market_cap_desc": ("market_cap", False),
    "turnover_desc": ("turnover", False),
}

OP_ALIASES = {
    "gt": ">",
    "greater_than": ">",
    "大于": ">",
    "gte": ">=",
    "min": ">=",
    "至少": ">=",
    "lt": "<",
    "less_than": "<",
    "小于": "<",
    "lte": "<=",
    "max": "<=",
    "不超过": "<=",
    "eq": "==",
    "equals": "==",
    "等于": "==",
}

BASE_COLUMN_CANDIDATES = {
    "code": ["代码", "证券代码", "symbol", "代码代码"],
    "name": ["名称", "证券简称", "股票简称", "name"],
    "latest_price": ["最新价", "最新", "现价", "收盘"],
    "pe": ["市盈率-动态", "市盈率(动态)", "市盈率", "PE", "动态市盈率"],
    "pe_ttm": ["市盈率TTM", "市盈率(TTM)", "PE(TTM)", "滚动市盈率"],
    "pb": ["市净率", "PB"],
    "market_cap": ["总市值", "市值"],
    "turnover": ["成交额"],
    "price_change_60d": ["60日涨跌幅", "60日涨跌幅%"],
    "ytd_change": ["年初至今涨跌幅", "年初至今涨跌幅%"],
}

HK_UNIVERSE = [
    "0001.HK",
    "0002.HK",
    "0003.HK",
    "0005.HK",
    "0016.HK",
    "0027.HK",
    "0066.HK",
    "0175.HK",
    "0267.HK",
    "0288.HK",
    "0386.HK",
    "0700.HK",
    "0762.HK",
    "0883.HK",
    "0939.HK",
    "0941.HK",
    "0960.HK",
    "0981.HK",
    "0992.HK",
    "1299.HK",
    "1810.HK",
    "2318.HK",
    "2388.HK",
    "3690.HK",
    "3988.HK",
    "9618.HK",
    "9633.HK",
    "9888.HK",
    "9988.HK",
]

DEFAULT_MAX_PROVIDER_CANDIDATES = 120


@dataclass(frozen=True)
class Filter:
    field: str
    op: str
    value: float


@dataclass(frozen=True)
class NormalizedCriteria:
    filters: list[Filter]
    exclude: list[str]
    sort_by: str | None
    limit: int
    warnings: list[str]


class MarketDataProvider(Protocol):
    def load_market(self, market: str, needed_fields: set[str] | None = None):
        ...


class CompositeMarketDataProvider:
    def __init__(
        self,
        *,
        a_share_provider: MarketDataProvider | None = None,
        hk_provider: MarketDataProvider | None = None,
    ) -> None:
        self.a_share_provider = a_share_provider or BaoStockAProvider()
        self.hk_provider = hk_provider or YFinanceHKProvider()

    def load_market(self, market: str, needed_fields: set[str] | None = None):
        if market == "A_SHARE":
            return self.a_share_provider.load_market(market, needed_fields=needed_fields)
        if market == "HK":
            return self.hk_provider.load_market(market, needed_fields=needed_fields)
        raise ValueError(f"Unsupported market: {market}")


class BaoStockAProvider:
    def __init__(self, *, max_candidates: int | None = None) -> None:
        self.max_candidates = max_candidates or _max_provider_candidates()

    def load_market(self, market: str, needed_fields: set[str] | None = None):
        _require_pandas()
        try:
            import baostock as bs
        except ImportError as exc:
            raise RuntimeError("baostock is required for A-share screening. Run pip install -r requirements.txt.") from exc

        login = bs.login()
        if login.error_code != "0":
            raise RuntimeError(f"BaoStock login failed: {login.error_msg}")

        try:
            universe = _baostock_recent_universe(bs)
            active_rows = [
                row
                for row in universe
                if _is_a_share_stock_code(row.get("code", ""))
                and row.get("tradeStatus", "1") == "1"
            ][: self.max_candidates]

            rows = []
            for item in active_rows:
                stock_row = _baostock_latest_row(
                    bs,
                    item["code"],
                    item.get("code_name"),
                    needed_fields=needed_fields,
                )
                if stock_row:
                    rows.append(stock_row)
            return _a_share_rows_to_frame(rows)
        finally:
            bs.logout()


class YFinanceHKProvider:
    def __init__(self, *, symbols: list[str] | None = None) -> None:
        self.symbols = symbols or HK_UNIVERSE

    def load_market(self, market: str, needed_fields: set[str] | None = None):
        _require_pandas()
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError("yfinance is required for HK screening. Run pip install -r requirements.txt.") from exc

        rows = []
        for symbol in self.symbols:
            row = _yfinance_hk_row(yf, symbol, needed_fields=needed_fields)
            if row:
                rows.append(row)
        return _rows_to_frame(rows, "HK")


class StockScreener:
    def __init__(self, provider: MarketDataProvider | None = None) -> None:
        self.provider = provider or CompositeMarketDataProvider()

    def screen(self, arguments: dict[str, Any]) -> dict[str, Any]:
        criteria = normalize_criteria(arguments)
        markets = _markets_from_argument(arguments.get("market"))
        warnings = list(criteria.warnings)
        data_sources: list[str] = []
        result_frames: list[Any] = []
        failed_markets: list[str] = []
        needed_fields = _needed_fields_from_criteria(criteria)

        for market in markets:
            try:
                frame = self.provider.load_market(market, needed_fields=needed_fields)
            except Exception as exc:  # noqa: BLE001 - tool should report provider failures
                warnings.append(f"{market} provider error: {exc}")
                failed_markets.append(market)
                continue

            data_sources.append(_data_source_for_market(market))
            filtered, filter_warnings = _apply_filters(frame, criteria.filters)
            warnings.extend(f"{market}: {warning}" for warning in filter_warnings)
            filtered = _apply_exclusions(filtered, criteria.exclude, warnings, market)
            filtered = _apply_sort(filtered, criteria.sort_by, warnings, market)
            result_frames.append(filtered)

        combined = _concat_frames(result_frames)
        if combined is not None and len(combined) > criteria.limit:
            combined = combined.head(criteria.limit)

        status = _status_for_result(combined, markets, failed_markets)
        return {
            "status": status,
            "market": arguments.get("market", "BOTH"),
            "criteria": _criteria_to_json(criteria),
            "results": _frame_to_results(combined),
            "warnings": warnings,
            "data_sources": data_sources,
        }


def _require_pandas() -> None:
    if pd is None:
        raise RuntimeError("pandas is required for stock_screener. Run pip install -r requirements.txt.")


def _baostock_rows(result: Any) -> list[dict[str, str]]:
    if result.error_code != "0":
        raise RuntimeError(result.error_msg)
    rows = []
    while result.next():
        rows.append(dict(zip(result.fields, result.get_row_data())))
    return rows


def _baostock_recent_universe(bs: Any) -> list[dict[str, str]]:
    for offset in range(0, 14):
        day = date.today() - timedelta(days=offset)
        rows = _baostock_rows(bs.query_all_stock(day=day.isoformat()))
        if rows:
            return rows
    raise RuntimeError("BaoStock returned no stock universe for the last 14 calendar days")


def _is_a_share_stock_code(code: str) -> bool:
    return code.startswith("sh.6") or code.startswith("sz.0") or code.startswith("sz.3")


def _baostock_latest_row(
    bs: Any,
    code: str,
    name: str | None = None,
    *,
    needed_fields: set[str] | None = None,
) -> dict[str, Any] | None:
    end = date.today()
    start = end - timedelta(days=140)
    fields = "date,code,close,volume,amount,turn,peTTM,pbMRQ"
    result = bs.query_history_k_data_plus(
        code,
        fields,
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        frequency="d",
        adjustflag="3",
    )
    rows = _baostock_rows(result)
    if not rows:
        return None

    latest = rows[-1]
    first = rows[0]
    ytd_row = _first_row_on_or_after(rows, date(end.year, 1, 1))
    latest_close = _to_number(latest.get("close"))
    first_close = _to_number(first.get("close"))
    ytd_close = _to_number(ytd_row.get("close")) if ytd_row else None
    needs_roe = needed_fields is not None and "roe" in needed_fields
    latest_roe = _baostock_latest_dupont_roe(bs, code) if needs_roe else None
    return {
        "market": "A_SHARE",
        "code": latest.get("code"),
        "name": name,
        "latest_price": latest_close,
        "pe": _to_number(latest.get("peTTM")),
        "pe_ttm": _to_number(latest.get("peTTM")),
        "pb": _to_number(latest.get("pbMRQ")),
        "dividend_yield": None,
        "roe": latest_roe,
        "market_cap": None,
        "turnover": _to_number(latest.get("amount")),
        "price_change_60d": _pct_change(first_close, latest_close),
        "ytd_change": _pct_change(ytd_close, latest_close),
    }


def _a_share_rows_to_frame(rows: list[dict[str, Any]]):
    return _rows_to_frame(rows, "A_SHARE")


def _baostock_latest_dupont_roe(bs: Any, code: str) -> float | None:
    for year, quarter in _recent_year_quarters(8):
        result = bs.query_dupont_data(code=code, year=year, quarter=quarter)
        rows = _baostock_rows(result)
        if not rows:
            continue
        value = _normalize_ratio_percent(rows[0].get("dupontROE"))
        if value is not None:
            return value
    return None


def _recent_year_quarters(limit: int) -> list[tuple[int, int]]:
    today = date.today()
    current_quarter = ((today.month - 1) // 3) + 1
    year = today.year
    quarter = current_quarter
    periods: list[tuple[int, int]] = []
    for _ in range(limit):
        periods.append((year, quarter))
        quarter -= 1
        if quarter == 0:
            year -= 1
            quarter = 4
    return periods


def _yfinance_hk_row(yf: Any, symbol: str, *, needed_fields: set[str] | None = None) -> dict[str, Any] | None:
    ticker = yf.Ticker(symbol)
    try:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            history = ticker.history(period="6mo", auto_adjust=False)
    except Exception:
        return None
    if history.empty:
        return None

    latest = history.iloc[-1]
    first = history.iloc[0]
    ytd = history[history.index.date >= date(date.today().year, 1, 1)]
    ytd_first = ytd.iloc[0] if not ytd.empty else None
    info_fields = {"pe", "pe_ttm", "pb", "dividend_yield", "roe", "market_cap"}
    needs_info = needed_fields is None or bool(needed_fields & info_fields)
    info = _safe_yfinance_info(ticker) if needs_info else {}
    fast_info = _safe_mapping(ticker.fast_info) if (needed_fields is None or "market_cap" in needed_fields) else {}
    latest_close = _to_number(latest.get("Close"))
    return {
        "market": "HK",
        "code": symbol,
        "name": info.get("shortName") or info.get("longName"),
        "latest_price": latest_close,
        "pe": _to_number(info.get("trailingPE") or info.get("forwardPE")),
        "pe_ttm": _to_number(info.get("trailingPE")),
        "pb": _to_number(info.get("priceToBook")),
        "dividend_yield": _normalize_dividend_yield(info.get("dividendYield")),
        "roe": _normalize_ratio_percent(info.get("returnOnEquity")),
        "market_cap": _to_number(fast_info.get("market_cap") or info.get("marketCap")),
        "turnover": _to_number(latest.get("Volume")),
        "price_change_60d": _pct_change(_to_number(first.get("Close")), latest_close),
        "ytd_change": _pct_change(_to_number(ytd_first.get("Close")) if ytd_first is not None else None, latest_close),
    }


def _safe_yfinance_info(ticker: Any) -> dict[str, Any]:
    try:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            info = ticker.info
    except Exception:
        return {}
    return info if isinstance(info, dict) else {}


def _safe_mapping(value: Any) -> dict[str, Any]:
    try:
        return dict(value)
    except Exception:
        return {}


def _rows_to_frame(rows: list[dict[str, Any]], market: str):
    _require_pandas()
    if not rows:
        return pd.DataFrame(columns=_output_fields())  # type: ignore[union-attr]
    frame = pd.DataFrame(rows)  # type: ignore[union-attr]
    frame["market"] = market
    for field in _output_fields():
        if field not in frame.columns:
            frame[field] = None
    numeric_fields = [field for field in _output_fields() if field not in {"market", "code", "name"}]
    for field in numeric_fields:
        frame[field] = frame[field].map(_to_number)
    return frame


def _first_row_on_or_after(rows: list[dict[str, str]], target: date) -> dict[str, str] | None:
    for row in rows:
        row_date = _parse_date(row.get("date"))
        if row_date and row_date >= target:
            return row
    return None


def _parse_date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def _pct_change(start: float | None, end: float | None) -> float | None:
    if start in (None, 0) or end is None:
        return None
    return ((end - start) / start) * 100


def _normalize_dividend_yield(value: Any) -> float | None:
    numeric = _to_number(value)
    if numeric is None:
        return None
    return numeric * 100 if 0 <= numeric <= 1 else numeric


def _normalize_ratio_percent(value: Any) -> float | None:
    numeric = _to_number(value)
    if numeric is None:
        return None
    return numeric * 100 if -1 <= numeric <= 1 else numeric


def _max_provider_candidates() -> int:
    raw = _to_number(os.environ.get("STOCK_SCREENER_MAX_PROVIDER_CANDIDATES"))
    if raw is None:
        return DEFAULT_MAX_PROVIDER_CANDIDATES
    return max(1, min(int(raw), 1000))


def _data_source_for_market(market: str) -> str:
    if market == "A_SHARE":
        return "baostock:A_SHARE:history_k_data_plus"
    if market == "HK":
        return "yfinance:HK:history_fast_info"
    return f"unknown:{market}"


def _status_for_result(combined: Any | None, markets: list[str], failed_markets: list[str]) -> str:
    if combined is None:
        return "error" if failed_markets else "ok"
    if failed_markets and len(failed_markets) < len(markets):
        return "partial"
    return "ok"


def _needed_fields_from_criteria(criteria: NormalizedCriteria) -> set[str]:
    needed_fields = {"latest_price"}
    for item in criteria.filters:
        needed_fields.add(item.field)
    if criteria.sort_by is not None:
        needed_fields.add(SORT_ALIASES[criteria.sort_by][0])
    if "recent_hot_high_valuation" in criteria.exclude:
        needed_fields.update({"pe", "price_change_60d"})
    return needed_fields


def normalize_criteria(arguments: dict[str, Any]) -> NormalizedCriteria:
    raw_criteria = arguments.get("criteria") or {}
    filters: list[Filter] = []
    warnings: list[str] = []

    if isinstance(raw_criteria.get("filters"), list):
        for raw_filter in raw_criteria["filters"]:
            parsed = _parse_filter(raw_filter)
            if parsed is None:
                warnings.append(f"Unsupported filter ignored: {raw_filter}")
            else:
                filters.append(parsed)

    filters.extend(_legacy_filters(raw_criteria, warnings))

    exclude = _normalize_string_list(raw_criteria.get("exclude") or arguments.get("exclude"))
    sort_by = raw_criteria.get("sort_by") or arguments.get("sort_by")
    if sort_by is not None and sort_by not in SORT_ALIASES:
        warnings.append(f"Unsupported sort_by ignored: {sort_by}")
        sort_by = None

    limit = _limit_from(arguments.get("limit") or raw_criteria.get("limit"))
    return NormalizedCriteria(
        filters=filters,
        exclude=exclude,
        sort_by=sort_by,
        limit=limit,
        warnings=warnings,
    )


def _normalize_market_frame(raw, market: str):
    frame = raw.copy()
    normalized = pd.DataFrame()  # type: ignore[union-attr]
    normalized["market"] = market

    for field, candidates in BASE_COLUMN_CANDIDATES.items():
        source = _first_present_column(frame, candidates)
        if source is None:
            normalized[field] = None
            continue
        normalized[field] = frame[source]

    numeric_fields = [
        "latest_price",
        "pe",
        "pe_ttm",
        "pb",
        "market_cap",
        "turnover",
        "price_change_60d",
        "ytd_change",
    ]
    for field in numeric_fields:
        normalized[field] = normalized[field].map(_to_number)

    normalized["dividend_yield"] = None
    normalized["roe"] = None
    return normalized


def _parse_filter(raw_filter: Any) -> Filter | None:
    if not isinstance(raw_filter, dict):
        return None

    field = _canonical_field(raw_filter.get("field"))
    op = OP_ALIASES.get(str(raw_filter.get("op", "")).lower(), raw_filter.get("op"))
    value = _to_number(raw_filter.get("value"))
    if field not in SUPPORTED_FIELDS or op not in {">", ">=", "<", "<=", "=="} or value is None:
        return None
    return Filter(field=field, op=str(op), value=float(value))


def _legacy_filters(criteria: dict[str, Any], warnings: list[str]) -> list[Filter]:
    filters: list[Filter] = []
    for key, value in criteria.items():
        if key in {"filters", "exclude", "sort_by", "limit"}:
            continue
        field, op = _legacy_key_to_filter(key)
        numeric_value = _to_number(value)
        if field is None or numeric_value is None:
            warnings.append(f"Unsupported criteria ignored: {key}")
            continue
        filters.append(Filter(field=field, op=op, value=float(numeric_value)))
    return filters


def _legacy_key_to_filter(key: str) -> tuple[str | None, str]:
    lowered = key.lower()
    for suffix, op in (("_min", ">="), ("_max", "<=")):
        if lowered.endswith(suffix):
            return _canonical_field(lowered[: -len(suffix)]), op
    return _canonical_field(lowered), "=="


def _canonical_field(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    lowered = raw.lower()
    return FIELD_ALIASES.get(lowered) or FIELD_ALIASES.get(raw) or lowered


def _apply_filters(frame, filters: list[Filter]) -> tuple[Any, list[str]]:
    warnings: list[str] = []
    filtered = frame
    for item in filters:
        if item.field not in filtered.columns:
            warnings.append(f"filter field unavailable: {item.field}")
            continue
        series = filtered[item.field].map(_to_number)
        comparable = series.notna()
        if not comparable.any():
            warnings.append(f"filter field has no available data: {item.field}")
            continue

        if item.op == ">":
            mask = series > item.value
        elif item.op == ">=":
            mask = series >= item.value
        elif item.op == "<":
            mask = series < item.value
        elif item.op == "<=":
            mask = series <= item.value
        else:
            mask = series == item.value
        filtered = filtered[mask.fillna(False)]
    return filtered, warnings


def _apply_exclusions(frame, exclude: list[str], warnings: list[str], market: str):
    filtered = frame
    if "recent_hot_high_valuation" not in exclude:
        return filtered

    columns = set(filtered.columns)
    if {"pe", "price_change_60d"} <= columns:
        pe = filtered["pe"].map(_to_number)
        change = filtered["price_change_60d"].map(_to_number)
        filtered = filtered[~((pe > 60) & (change > 30)).fillna(False)]
    else:
        warnings.append(f"{market}: recent_hot_high_valuation exclusion skipped because PE or 60-day change is unavailable")
    return filtered


def _apply_sort(frame, sort_by: str | None, warnings: list[str], market: str):
    if sort_by is None:
        return frame
    field, ascending = SORT_ALIASES[sort_by]
    if field not in frame.columns:
        warnings.append(f"{market}: sort field unavailable: {field}")
        return frame
    return frame.sort_values(by=field, ascending=ascending, na_position="last")


def _markets_from_argument(value: Any) -> list[str]:
    if value == "A_SHARE":
        return ["A_SHARE"]
    if value == "HK":
        return ["HK"]
    return ["A_SHARE", "HK"]


def _limit_from(value: Any) -> int:
    numeric = _to_number(value)
    if numeric is None:
        return 20
    return max(1, min(int(numeric), 100))


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _first_present_column(frame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def _concat_frames(frames: list[Any]) -> Any | None:
    non_empty = [frame for frame in frames if frame is not None and not frame.empty]
    if not non_empty:
        return None
    compact = [frame.dropna(axis=1, how="all") for frame in non_empty]
    combined = pd.concat(compact, ignore_index=True)  # type: ignore[union-attr]
    for field in _output_fields():
        if field not in combined.columns:
            combined[field] = None
    return combined


def _frame_to_results(frame) -> list[dict[str, Any]]:
    if frame is None:
        return []

    results: list[dict[str, Any]] = []
    for row in frame[_output_fields()].to_dict(orient="records"):
        results.append({key: _json_safe(value) for key, value in row.items()})
    return results


def _output_fields() -> list[str]:
    return [
        "market",
        "code",
        "name",
        "latest_price",
        "pe",
        "pe_ttm",
        "pb",
        "dividend_yield",
        "roe",
        "market_cap",
        "turnover",
        "price_change_60d",
        "ytd_change",
    ]


def _criteria_to_json(criteria: NormalizedCriteria) -> dict[str, Any]:
    return {
        "filters": [
            {"field": item.field, "op": item.op, "value": item.value}
            for item in criteria.filters
        ],
        "exclude": criteria.exclude,
        "sort_by": criteria.sort_by,
        "limit": criteria.limit,
    }


def _to_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)

    text = str(value).strip()
    if not text or text in {"-", "--", "nan", "None"}:
        return None
    multiplier = 1.0
    if text.endswith("%"):
        text = text[:-1]
    if text.endswith("亿"):
        multiplier = 100_000_000.0
        text = text[:-1]
    elif text.endswith("万"):
        multiplier = 10_000.0
        text = text[:-1]
    text = text.replace(",", "")
    try:
        return float(text) * multiplier
    except ValueError:
        return None


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd is not None and pd.isna(value):
        return None
    return value
