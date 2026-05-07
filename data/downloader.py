"""
Módulo de descarga y caché de datos financieros.
Fuente principal: Yahoo Finance (yfinance).
Fallback: Alpha Vantage (requiere API key en .env).
"""

import os
import json
import hashlib
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

ASSETS_CONFIG_PATH = Path(__file__).parent.parent / "assets" / "watchlist.json"

# ── Fuentes de datos ──────────────────────────────────────────────────────────

def download_yfinance(tickers: list[str], start: str, end: str, interval: str = "1d") -> dict:
    """Descarga datos OHLCV desde Yahoo Finance."""
    results = {}
    failed = []
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end, interval=interval,
                             auto_adjust=True, progress=False)
            if df.empty:
                failed.append(ticker)
            else:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                df.index = pd.to_datetime(df.index)
                df.index = df.index.tz_localize(None)
                results[ticker] = df
        except Exception as e:
            failed.append(ticker)
            print(f"[ERROR] {ticker}: {e}")
    return results, failed


def download_alpha_vantage(ticker: str, api_key: str, outputsize: str = "full") -> pd.DataFrame:
    """
    Alternativa: Alpha Vantage (TIME_SERIES_DAILY_ADJUSTED).
    Requiere AV_API_KEY en .env
    """
    import requests
    url = (
        f"https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_DAILY_ADJUSTED"
        f"&symbol={ticker}&outputsize={outputsize}"
        f"&apikey={api_key}"
    )
    r = requests.get(url, timeout=10)
    data = r.json()
    ts = data.get("Time Series (Daily)", {})
    if not ts:
        raise ValueError(f"Alpha Vantage no devolvió datos para {ticker}")
    df = pd.DataFrame(ts).T
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.rename(columns={
        "1. open": "Open", "2. high": "High", "3. low": "Low",
        "4. close": "Close", "5. adjusted close": "Adj Close",
        "6. volume": "Volume"
    })
    df = df.astype(float)
    return df


# ── Caché local ───────────────────────────────────────────────────────────────

def _cache_key(tickers: list, start: str, end: str, interval: str) -> str:
    raw = f"{'_'.join(sorted(tickers))}_{start}_{end}_{interval}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.parquet"


def load_prices_cached(tickers: list[str], start: str, end: str,
                        interval: str = "1d", force_refresh: bool = False) -> tuple:
    """
    Descarga precios con caché inteligente.
    Reusa datos guardados si son recientes (< 1 día).
    """
    key = _cache_key(tickers, start, end, interval)
    path = _cache_path(key)

    if not force_refresh and path.exists():
        age_hours = (datetime.now().timestamp() - path.stat().st_mtime) / 3600
        if age_hours < 24:
            df = pd.read_parquet(path)
            failed = [t for t in tickers if t not in df.columns.get_level_values(0)]
            return df, failed, True  # True = from cache

    raw, failed = download_yfinance(tickers, start, end, interval)
    if not raw:
        return pd.DataFrame(), tickers, False

    # Construir DataFrame multi-nivel (ticker, campo)
    frames = {}
    for ticker, df in raw.items():
        frames[ticker] = df

    # Solo quedarnos con Close para optimización
    close_data = {}
    for ticker, df in frames.items():
        if "Close" in df.columns:
            close_data[ticker] = df["Close"]

    combined = pd.DataFrame(close_data)
    combined = combined.ffill().dropna(how="all")
    combined.to_parquet(path)

    return combined, failed, False


# ── Info de activos ───────────────────────────────────────────────────────────

def get_ticker_info(ticker: str) -> dict:
    """Obtiene metadata de un activo: nombre, sector, moneda, etc."""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        return {
            "ticker": ticker,
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "currency": info.get("currency", "USD"),
            "country": info.get("country", "N/A"),
            "market_cap": info.get("marketCap", None),
            "beta": info.get("beta", None),
            "dividend_yield": info.get("dividendYield", None),
            "pe_ratio": info.get("trailingPE", None),
            "52w_high": info.get("fiftyTwoWeekHigh", None),
            "52w_low": info.get("fiftyTwoWeekLow", None),
            "analyst_target": info.get("targetMeanPrice", None),
            "analyst_low": info.get("targetLowPrice", None),
            "analyst_high": info.get("targetHighPrice", None),
            "analyst_count": info.get("numberOfAnalystOpinions", None),
            "current_price": info.get("currentPrice", info.get("regularMarketPrice", None)),
            "description": info.get("longBusinessSummary", "")[:300] if info.get("longBusinessSummary") else "",
        }
    except Exception as e:
        return {"ticker": ticker, "name": ticker, "error": str(e)}


# ── Configuración de watchlist ────────────────────────────────────────────────

def load_watchlist() -> list:
    if ASSETS_CONFIG_PATH.exists():
        with open(ASSETS_CONFIG_PATH) as f:
            return json.load(f)
    return [
        {"ticker": "SPY", "name": "S&P 500 ETF", "type": "ETF"},
        {"ticker": "QQQ", "name": "Nasdaq 100 ETF", "type": "ETF"},
        {"ticker": "AGG", "name": "Bond Aggregate ETF", "type": "Bond"},
        {"ticker": "GLD", "name": "Gold ETF", "type": "Commodity"},
    ]


def save_watchlist(assets: list):
    ASSETS_CONFIG_PATH.parent.mkdir(exist_ok=True)
    with open(ASSETS_CONFIG_PATH, "w") as f:
        json.dump(assets, f, indent=2)
