from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats as sps

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RISK_FREE_RATE = 0.0526       # fallback; overwritten by ^IRX at runtime
AAA_YIELD_PCT = 5.78         # Graham formula reference yield, requires updating according to market
MARKET_PREMIUM = 0.042        # equity risk premium

GEMINI_MODEL = "gemini-3.5-flash-lite"


PERIODS_PER_YEAR = {
    "1m": 252 * 390, "2m": 252 * 195, "5m": 252 * 78, "15m": 252 * 26,
    "30m": 252 * 13, "60m": 252 * 6.5, "90m": 252 * 6.5 * 60 / 90, "1h": 252 * 6.5,
    "1d": 252, "5d": 252 / 5, "1wk": 52, "1mo": 12, "3mo": 4,
}


# Data collection from yfinance
def get_ohlcv(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data for '{ticker}'")
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = df.index.tz_localize(None)
    return df


def get_fundamentals(ticker: str) -> dict:
    info = yf.Ticker(ticker).get_info()
    keys = [
        "sector", "industry", "trailingPE", "forwardPE", "priceToBook",
        "returnOnEquity", "profitMargins", "revenueGrowth", "debtToEquity",
        "dividendYield", "marketCap", "beta",
        "trailingEps", "forwardEps", "earningsGrowth", "currentPrice",
    ]
    return {k: info.get(k) for k in keys}


def refresh_risk_free() -> float:
    """Latest 13-week T-bill yield (^IRX, quoted in %); keeps the fallback on failure."""
    global RISK_FREE_RATE
    try:
        RISK_FREE_RATE = float(yf.Ticker("^IRX").history(period="5d")["Close"].dropna().iloc[-1]) / 100
    except Exception as e:
        print(f"[rf] ^IRX unavailable ({e}); using {RISK_FREE_RATE:.2%}", file=sys.stderr)
    return RISK_FREE_RATE


# Stats & risk
def log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices / prices.shift(1)).dropna()


def annualized_volatility(returns: pd.Series, periods: int) -> float:
    return float(returns.std() * np.sqrt(periods))


def sharpe_ratio(returns: pd.Series, periods: int) -> float:
    excess = returns - RISK_FREE_RATE / periods
    return float(np.sqrt(periods) * excess.mean() / excess.std())


def max_drawdown(prices: pd.Series) -> tuple[float, pd.Series]:
    dd = prices / prices.cummax() - 1
    return float(dd.min()), dd


def historical_var(returns: pd.Series, level: float = 0.95) -> float:
    return float(-np.percentile(returns, (1 - level) * 100))


def historical_cvar(returns: pd.Series, level: float = 0.95) -> float:
    tail = returns[returns <= np.percentile(returns, (1 - level) * 100)]
    return float(-tail.mean()) if len(tail) else float("nan")


def beta_alpha(asset: pd.Series, bench: pd.Series, periods: int):
    """Jensen's alpha on excess returns (proper CAPM)."""
    rf_pp = RISK_FREE_RATE / periods
    slope, intercept, r, _, _ = sps.linregress(bench - rf_pp, asset - rf_pp)
    return float(slope), float(intercept * periods), float(r ** 2)


def hurst_exponent(prices: pd.Series, max_lag: int = 100) -> float:
    log_p = np.log(prices.values)
    lags = range(2, min(max_lag, len(log_p) // 2))
    tau = [np.std(log_p[lag:] - log_p[:-lag]) for lag in lags]
    return float(np.polyfit(np.log(list(lags)), np.log(tau), 1)[0])


def sma(prices: pd.Series, window: int) -> pd.Series:
    return prices.rolling(window).mean()


def _best_ar_lags(y: np.ndarray, max_lag: int = 5) -> int:
    """BIC-selected AR order for the return mean (0 = constant mean)."""
    from arch import arch_model
    bic = {0: arch_model(y, mean="Constant", vol="Constant", hold_back=max_lag).fit(disp="off").bic}
    for k in range(1, max_lag + 1):
        bic[k] = arch_model(y, mean="AR", lags=k, vol="Constant", hold_back=max_lag).fit(disp="off").bic
    return min(bic, key=bic.get)


def garch_model(returns: pd.Series, periods: int, horizon: int) -> dict:
    """AR(p)-EGARCH(1,1,1) Student-t: gives mean and vol paths."""
    from arch import arch_model
    y = returns.to_numpy() * 100
    lags = _best_ar_lags(y)
    mean_kw = {"mean": "AR", "lags": lags} if lags else {"mean": "Constant"}
    res = arch_model(y, vol="EGARCH", p=1, o=1, q=1, dist="t", **mean_kw).fit(disp="off")
    fc = res.forecast(horizon=horizon, method="simulation", simulations=1000, reindex=False)
    fc_var = fc.variance.values[-1]
    fc_vol = np.sqrt(fc_var * periods) / 100
    return {
        "persistence": float(res.params["beta[1]"]),
        "fc_ann_vol": float(fc_vol[0]),
        "fc_path": fc_vol,
        "mean_lr": fc.mean.values[-1] / 100,           # expected log return per step
        "cum_sd": np.sqrt(np.cumsum(fc_var)) / 100,    # horizon log-return std at each step
    }


# Fair value = three-pillar model + reverse DCF
def graham_value(eps: float, growth_pct: float) -> float:
    """Graham: V = EPS x (8.5 + 2g) x (4.4 / Y)."""
    g = float(np.clip(growth_pct, -5.0, 25.0))
    return eps * (8.5 + 2.0 * g) * (4.4 / AAA_YIELD_PCT)


def earnings_power_value(eps: float, growth_pct: float, discount: float,
                         years: int = 5, terminal_g: float = 0.025) -> float:
    """Two-stage earnings DCF: growth stage, then Gordon terminal."""
    g = float(np.clip(growth_pct / 100.0, -0.10, 0.25))
    r = discount
    value, earnings = 0.0, eps
    for t in range(1, years + 1):
        earnings *= 1.0 + g
        value += earnings / (1.0 + r) ** t
    terminal = earnings * (1.0 + terminal_g) / (r - terminal_g)
    return value + terminal / (1.0 + r) ** years


def multiple_reversion_value(forward_eps: float, historical_pe: float) -> float:
    pe = float(np.clip(historical_pe, 8.0, 30.0))
    return forward_eps * pe


def implied_growth(price: float, eps: float, discount: float) -> float:
    """Reverse DCF: growth rate (in %) that makes DCF value == market price."""
    lo, hi = -10.0, 30.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        if earnings_power_value(eps, mid, discount) < price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def fair_value_gf(fundamentals: dict, beta: float) -> dict | None:
    """Graham (35%) + earnings DCF (40%) + multiple reversion (25%)."""
    eps = fundamentals.get("trailingEps")
    price = fundamentals.get("currentPrice")
    if not eps or eps <= 0 or not price:
        return None

    growth_src = fundamentals.get("earningsGrowth") or fundamentals.get("revenueGrowth") or 0.05
    growth_pct = float(growth_src) * 100.0
    forward_eps = fundamentals.get("forwardEps")
    if not forward_eps or forward_eps <= 0:
        forward_eps = eps
    hist_pe = fundamentals.get("trailingPE") or 15.0
    cost_of_equity = RISK_FREE_RATE + max(float(beta or 1.0), 0.5) * MARKET_PREMIUM

    graham = graham_value(eps, growth_pct)
    dcf = earnings_power_value(eps, growth_pct, cost_of_equity)
    reversion = multiple_reversion_value(forward_eps, hist_pe)

    blend = 0.35 * graham + 0.40 * dcf + 0.25 * reversion
    margin_of_safety = (blend - price) / price
    implied = implied_growth(price, eps, cost_of_equity)

    if margin_of_safety > 0.20:
        verdict, tag = "UNDERVALUED", "up"
    elif margin_of_safety < -0.20:
        verdict, tag = "OVERVALUED", "down"
    else:
        verdict, tag = "FAIRLY VALUED", "flat"

    return {
        "price": price, "graham": graham, "dcf": dcf, "reversion": reversion,
        "fair_value": blend, "margin_of_safety": margin_of_safety,
        "implied_growth_pct": implied, "consensus_growth_pct": growth_pct,
        "cost_of_equity": cost_of_equity, "verdict": verdict, "tag": tag,
    }


# Forecast engines : AR-GARCH default, Kronos opt-in
def _future_timestamps(index: pd.DatetimeIndex, n: int) -> pd.DatetimeIndex:
    step = index.to_series().diff().dropna().median() if len(index) > 1 else pd.Timedelta(days=1)
    if step >= pd.Timedelta(days=1):
        return pd.bdate_range(index[-1] + pd.Timedelta(days=1), periods=n)
    return pd.DatetimeIndex([index[-1] + step * (i + 1) for i in range(n)])


def ar_garch_forecast(df: pd.DataFrame, garch: dict) -> dict:
    """Expected path from the AR-GARCH fit; 5-95% band from its horizon variance."""
    base = float(df["Close"].iloc[-1])
    future = _future_timestamps(df.index, len(garch["mean_lr"]))
    cum = np.cumsum(garch["mean_lr"])
    band = 1.645 * garch["cum_sd"]
    return {
        "engine": "ar-garch",
        "mean_path": pd.Series(base * np.exp(cum), index=future),
        "lower": pd.Series(base * np.exp(cum - band), index=future),
        "upper": pd.Series(base * np.exp(cum + band), index=future),
    }


def _kronos_real(df, pred_len, n_samples, model_size) -> dict:
    import torch
    kronos_path = os.environ.get("KRONOS_DIR")
    if not kronos_path:
        raise RuntimeError("KRONOS_DIR env var not set")
    sys.path.append(kronos_path)
    from model import Kronos, KronosPredictor, KronosTokenizer # type: ignore

    tok, name = {
        "mini": ("NeoQuasar/Kronos-Tokenizer-2k", "NeoQuasar/Kronos-mini"),
        "small": ("NeoQuasar/Kronos-Tokenizer-base", "NeoQuasar/Kronos-small"),
        "base": ("NeoQuasar/Kronos-Tokenizer-base", "NeoQuasar/Kronos-base")
    }[model_size]

    tokenizer = KronosTokenizer.from_pretrained(tok)
    model = Kronos.from_pretrained(name)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

    ctx = df.tail(252)
    x_df = ctx.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].reset_index(drop=True)
    x_ts = pd.Series(ctx.index)
    y_ts = pd.Series(_future_timestamps(ctx.index, pred_len))

    preds = predictor.predict_batch(
        df_list=[x_df] * n_samples,
        x_timestamp_list=[x_ts] * n_samples,
        y_timestamp_list=[y_ts] * n_samples,
        pred_len=pred_len, T=0.2, top_p=0.98, sample_count=1, verbose=False,
    )
    paths = np.array([p["close"].values for p in preds])
    return {
        "engine": "kronos",
        "mean_path": pd.Series(paths.mean(axis=0), index=y_ts.values),
        "lower": pd.Series(np.percentile(paths, 5, axis=0), index=y_ts.values),
        "upper": pd.Series(np.percentile(paths, 95, axis=0), index=y_ts.values),
    }


# News & Gemini
def _gemini_client():
    from google import genai
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY env var not set")
    return genai.Client(api_key=key)


def _fetch_article(url: str, max_chars: int = 1400) -> str:
    try:
        import requests
        from bs4 import BeautifulSoup
        resp = requests.get(url, timeout=6, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
            tag.decompose()
        text = " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))
        return text[:max_chars]
    except Exception:
        return ""


def get_news(ticker: str, limit: int = 12, use_sentiment: bool = True,
             fetch_bodies: int = 6) -> tuple[list[dict], str]:
    try:
        raw = yf.Search(ticker, news_count=limit).news
    except Exception:
        raw = []

    articles = []
    for item in raw or []:
        content = item.get("content") or {}
        article = {
            "headline": content.get("title") or item.get("title") or "",
            "url": (content.get("canonicalUrl") or {}).get("url") or item.get("link") or "",
            "source": (content.get("provider") or {}).get("displayName") or item.get("publisher") or "",
            "summary": content.get("summary") or "",
        }
        if article["headline"]:
            articles.append(article)

    if not articles:
        return [], "no_news"

    for a in articles[:fetch_bodies]:
        body = a["summary"] or _fetch_article(a["url"])
        a["text"] = (a["headline"] + "\n" + body).strip()[:1800]
    for a in articles[fetch_bodies:]:
        a["text"] = a["headline"]

    if not use_sentiment:
        return [dict(a, sentiment=None) for a in articles], "disabled"

    try:
        numbered = "\n\n".join(f"[{i}] {a['text']}" for i, a in enumerate(articles))
        prompt = (
            f"Score each full news item for its impact on {ticker} specifically, -1.0..+1.0 "
            "(0 = neutral/macro). Calibration: beat+raise=0.8, miss+cut=-0.9, routine note=0.0, "
            f"sector selloff=-0.2. Reply ONLY with a JSON array of {len(articles)} numbers."
            f"\n\n{numbered}"
        )
        resp = _gemini_client().models.generate_content(model=GEMINI_MODEL, contents=prompt)
        txt = resp.text.strip().strip("`").removeprefix("json").strip()
        scores = [float(s) for s in json.loads(txt)]
        if len(scores) != len(articles):
            raise ValueError("score count mismatch")
        engine = "gemini"
    except Exception as e:
        print(f"[news] sentiment unavailable ({e})", file=sys.stderr)
        scores, engine = [None] * len(articles), "unavailable"

    return [dict(a, sentiment=s) for a, s in zip(articles, scores)], engine


def sentiment_summary(news: list[dict]) -> dict | None:
    scores = [n["sentiment"] for n in news if n.get("sentiment") is not None]
    if not scores:
        return None
    positive = sum(1 for s in scores if s > 0.3)
    negative = sum(1 for s in scores if s < 0)
    return {"avg": float(np.mean(scores)), "pos": positive,
            "neg": negative, "neu": len(scores) - positive - negative}


def ai_verdict(ticker: str, stats: dict, funds: dict, forecast: dict, use_verdict: bool,
               news: list[dict]) -> dict | None:
    """Context-engineered buy/no-buy call over stats + fundamentals + forecast + news."""
    if not use_verdict:
        return None
    try:
        client = _gemini_client()
    except Exception:
        return None

    top = sorted([n for n in news if n.get("sentiment") is not None],
                 key=lambda n: -abs(n["sentiment"]))[:8]
    news_block = "\n".join(
        f"- [{n['sentiment']:+.2f}] {n['headline']} :: {(n.get('summary') or '')[:160]}"
        for n in top) or "(none)"

    beta_key = "beta_" + stats["benchmark"]
    context = f"""TICKER {ticker} as of {pd.Timestamp.now():%Y-%m-%d}, price {stats['last_price']:.2f}
RISK ann_vol {stats['ann_vol_pct']}% sharpe {stats['sharpe']} maxDD {stats['max_dd_pct']}% VaR95 {stats['var95_pct']}% garch_fwd_vol {stats['garch_fc_vol_pct']}% beta {stats.get(beta_key)} hurst {stats['hurst']}
EDGE {len(forecast['mean_path'])}d expected {stats['exp_move_pct']:+.2f}% band {forecast['lower'].iloc[-1]:.2f}-{forecast['upper'].iloc[-1]:.2f}
FUNDS {json.dumps({k: v for k, v in funds.items() if v is not None}, default=str)}
NEWS avg {stats.get('news_avg')} :: {news_block}"""

    prompt = (
        """You are a skeptical senior portfolio manager working at Jane Street. Decide if this is a good buy NOW.
        Reply with ONLY this JSON (no markdown):\n
        {"rec":"strong_buy|buy|hold|reduce|avoid","conv":0.0,"horizon":"","thesis":"",
        "bull":["","",""],"bear":["","",""],"risks":["","",""],"flip":""}. Here are the information on the stock: \n\n""" + context
    )
    try:
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        txt = resp.text.strip().strip("`").removeprefix("json").strip()
        return json.loads(txt)
    except Exception as e:
        print(f"[verdict] failed ({e})", file=sys.stderr)
        return None


# Charts
def _finish(fig, ax, title, legend=False):
    if legend:
        ax.legend(fontsize=8)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    plt.close(fig)
    return fig


def build_market_figure(df: pd.DataFrame, ticker: str, forecast: dict, garch: dict, ppy: int):
    """Price + SMA50 + forecast band, with realized vs GARCH-forward vol below."""
    fig, (ax, ax_vol) = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})
    n = len(df)
    x = np.arange(n)
    x_f = np.arange(n, n + len(forecast["mean_path"]))
    ax.plot(x, df["Close"].to_numpy(), color="black", linewidth=1, label="close")
    ax.plot(x, sma(df["Close"], 50), label="SMA50")
    ax.plot(x_f, forecast["mean_path"].values, linestyle="--", label=f"forecast ({forecast['engine']})")
    ax.fill_between(x_f, forecast["lower"].values, forecast["upper"].values, alpha=0.2, label="5-95% band")
    ax.set_title(ticker)
    realized = np.log(df["Close"] / df["Close"].shift(1)).rolling(20).std() * np.sqrt(ppy) * 100
    ax_vol.plot(x, realized.to_numpy(), label="realized vol 20d")
    ax_vol.plot(x_f, garch["fc_path"] * 100, linestyle=":", label="GARCH forward vol")
    ax_vol.set_ylabel("ann vol %")
    for a in (ax, ax_vol):
        a.grid(alpha=0.25)
        a.legend(fontsize=8)
    step = max(1, n // 8)
    ax_vol.set_xticks(x[::step])
    ax_vol.set_xticklabels(df.index[::step].strftime("%Y-%m-%d"), rotation=30, ha="right")
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_correlation(corr: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)), corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, shrink=0.8)
    return _finish(fig, ax, "Return Correlation")


def plot_weights(weights: dict, title: str):
    items = {k: v for k, v in weights.items() if v > 0.001}
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(list(items), list(items.values()))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return _finish(fig, ax, title)

# Portfolio (HRP / Markowitz max-Sharpe)
def optimize_portfolio(tickers: list[str], period: str = "2y", method: str = "hrp") -> dict:
    from pypfopt import EfficientFrontier, expected_returns, risk_models
    from pypfopt.base_optimizer import portfolio_performance
    from pypfopt.hierarchical_portfolio import HRPOpt
    refresh_risk_free()
    prices = pd.DataFrame({t: get_ohlcv(t, period)["Close"] for t in tickers}).dropna()
    returns = np.log(prices / prices.shift(1)).dropna()

    mu = expected_returns.mean_historical_return(prices)
    cov = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    if method == "markowitz":
        ef = EfficientFrontier(mu, cov)
        ef.max_sharpe(risk_free_rate=RISK_FREE_RATE)
        weights = ef.clean_weights()
    else:
        weights = HRPOpt(returns).optimize()
    performance = portfolio_performance(weights, mu, cov, risk_free_rate=RISK_FREE_RATE)

    return {
        "weights": weights,
        "performance": performance,
        "method": method,
        "figures": {
            "weights": plot_weights(weights, "HRP Weights" if method == "hrp" else "Max-Sharpe Weights"),
            "correlation": plot_correlation(returns.corr()),
        },
    }

# Main analysis pipeline (very customizable just define functions above and inclue them in here and run all in 1 go)
def analyze(ticker: str, benchmark: str = "SPY", period: str = "2y",
            interval: str = "1d", pred_len: int = 21,
            use_kronos: bool = False, use_sentiment: bool = True, use_verdict: bool = True) -> dict:
    refresh_risk_free()
    df = get_ohlcv(ticker, period, interval)
    bench = get_ohlcv(benchmark, period, interval)
    ppy = PERIODS_PER_YEAR.get(interval, 252)
    close = df["Close"]
    returns = log_returns(close)
    aligned = pd.concat([returns, log_returns(bench["Close"])], axis=1, join="inner")
    beta, _, _ = beta_alpha(aligned.iloc[:, 0], aligned.iloc[:, 1], ppy)
    dd, _ = max_drawdown(close)
    var95, cvar95 = historical_var(returns), historical_cvar(returns)
    garch = garch_model(returns, ppy, pred_len)

    # forecast: AR-GARCH default, Kronos opt-in
    forecast = ar_garch_forecast(df, garch)
    if use_kronos:
        try:
            forecast = _kronos_real(df, pred_len, 5, "base")
        except Exception as e:
            print(f"[kronos] unavailable ({e}); keeping AR-GARCH.", file=sys.stderr)

    last = float(close.iloc[-1])
    mu = float(np.log(forecast["mean_path"].iloc[-1] / last))
    sigma = float(np.log(forecast["upper"].iloc[-1] / forecast["lower"].iloc[-1]) / (2 * 1.645))

    stats = {
        "ticker": ticker, "benchmark": benchmark, "last_price": last,
        "ann_vol_pct": round(annualized_volatility(returns, ppy) * 100, 2),
        "sharpe": round(sharpe_ratio(returns, ppy), 3),
        "max_dd_pct": round(dd * 100, 2),
        "var95_pct": round(var95 * 100, 2),
        "cvar95_pct": round(cvar95 * 100, 2),
        "garch_fc_vol_pct": round(float(garch["fc_ann_vol"]) * 100, 2),
        f"beta_{benchmark}": round(beta, 3),
        "hurst": round(hurst_exponent(close), 3),
        "exp_move_pct": round((float(forecast["mean_path"].iloc[-1]) / last - 1) * 100, 2),
    }

    fundamentals = get_fundamentals(ticker)
    fair_value = fair_value_gf(fundamentals, beta)

    news, news_engine = get_news(ticker, use_sentiment=use_sentiment)
    summary = sentiment_summary(news)
    if summary:
        stats["news_avg"] = round(summary["avg"], 3)
    verdict = ai_verdict(ticker, stats, fundamentals, forecast, use_verdict, news) if news_engine == "gemini" else None

    return {
        "stats": stats, "forecast": forecast, "fair_value": fair_value,
        "kelly": {"mu": mu, "sigma": sigma, "rf": RISK_FREE_RATE * pred_len / ppy},
        "news": news, "news_engine": news_engine,
        "verdict": verdict,
        "figure": build_market_figure(df, ticker, forecast, garch, ppy),
    }

# Terminal UI
def run_ui():
    import queue
    import threading
    import tkinter as tk
    from tkinter import messagebox, ttk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    root = tk.Tk()

    top = ttk.Frame(root)
    top.pack(fill="x", padx=8, pady=(8, 0))
    tabs = ttk.Notebook(root)
    tabs.pack(fill="both", expand=True, padx=8, pady=8)

    def entry_row(parent, specs):
        out = []
        for col, (label, default) in enumerate(specs):
            ttk.Label(parent, text=label).grid(row=0, column=2 * col, padx=(8, 2))
            var = tk.StringVar(value=str(default))
            ttk.Entry(parent, textvariable=var, width=10).grid(row=0, column=2 * col + 1, padx=(0, 8))
            out.append(var)
        return out

    def scroll_host(parent):
        """Scrollable wrapper so the tall market figure can be read at native size."""
        outer = ttk.Frame(parent)
        cv = tk.Canvas(outer, highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient="vertical", command=cv.yview)
        hsb = ttk.Scrollbar(outer, orient="horizontal", command=cv.xview)
        inner = ttk.Frame(cv)
        inner.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.create_window((0, 0), window=inner, anchor="nw")
        cv.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        cv.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)

        def wheel(ev):
            cv.yview_scroll(-1 if ev.delta > 0 else 1, "units")
        cv.bind("<Enter>", lambda e: cv.bind_all("<MouseWheel>", wheel))
        cv.bind("<Leave>", lambda e: cv.unbind_all("<MouseWheel>"))
        return outer, inner

    def show_figures(host, figs):
        for child in host.winfo_children():
            child.destroy()
        for fig in figs:
            canvas = FigureCanvasTkAgg(fig, master=host)
            canvas.get_tk_widget().pack()
            NavigationToolbar2Tk(canvas, host)
            canvas.draw()

    # ================= ticker =================
    t1 = ttk.Frame(tabs)
    tabs.add(t1, text="Ticker")
    c1 = ttk.Frame(t1)
    c1.pack(fill="x", pady=(8, 4))
    tv, bv, pv, hv = entry_row(c1, [("Ticker", "AAPL"), ("Benchmark", "SPY"), ("Period", "2y"), ("Horizon", 21)])
    iv = tk.StringVar(value="1d")
    ttk.Combobox(c1, textvariable=iv, state="readonly", width=6,
                 values=["1m", "2m", "5m", "15m", "30m", "1h", "90m", "1d", "5d", "1wk", "1mo"]
                 ).grid(row=0, column=8, padx=(0, 8))
    kv = tk.BooleanVar(value=False)
    ttk.Checkbutton(c1, text="Kronos", variable=kv).grid(row=0, column=9, padx=8)
    sv = tk.BooleanVar(value=True)
    ttk.Checkbutton(c1, text="AI analysis", variable=sv).grid(row=0, column=10, padx=8)
    ttk.Button(c1, text="Analyze", command=lambda: run_ticker()).grid(row=0, column=11, padx=8)

    body1 = ttk.Frame(t1)
    body1.pack(fill="both", expand=True)
    summary = tk.Text(body1, width=56, wrap="word", font=("Courier", 10))
    summary.pack(side="left", fill="y")
    chart_outer, chart_host = scroll_host(body1)
    chart_outer.pack(side="left", fill="both", expand=True, padx=(8, 0))
    news = tk.Text(t1, height=14, wrap="word", font=("Courier", 9))
    news.pack(fill="x", pady=(8, 0))

    def render_ticker(res: dict):
        s, k, fv, vd, fc = res["stats"], res["kelly"], res["fair_value"], res["verdict"], res["forecast"]
        beta = s[f"beta_{s['benchmark']}"]
        f = float(np.clip((k["mu"] + k["sigma"] ** 2 / 2 - k["rf"]) / max(k["sigma"] ** 2, 1e-9), -0.30, 0.30))
        lines = [
            f"{s['ticker']}  {s['last_price']:.2f}  [{fc['engine']}]",
            "",
            f"FORECAST {len(fc['mean_path'])}d  mu {k['mu']:+.2%}  sigma {k['sigma']:.2%}",
            f"  5-95% band  {fc['lower'].iloc[-1]:.2f} .. {fc['upper'].iloc[-1]:.2f}",
            f"SIZING  kelly {f:+.1%}  half {f / 2:+.1%}  quarter {f / 4:+.1%}  (rf {RISK_FREE_RATE:.2%})",
            "",
            "RISK",
            f"  ann vol {s['ann_vol_pct']}%  garch fwd {s['garch_fc_vol_pct']}%",
            f"  VaR95 {s['var95_pct']}%  CVaR95 {s['cvar95_pct']}%  maxDD {s['max_dd_pct']}%",
            f"  sharpe {s['sharpe']}  beta {beta}  hurst {s['hurst']}",
            "",
            "FAIR VALUE",
        ]
        if fv:
            lines += [
                f"  {fv['verdict']}  blend {fv['fair_value']:.2f} vs {fv['price']:.2f}  MoS {fv['margin_of_safety']:+.1%}",
                f"  graham {fv['graham']:.2f}  dcf {fv['dcf']:.2f}  reversion {fv['reversion']:.2f}",
            ]
        else:
            lines.append("  n/a (no EPS)")
        lines += ["", "AI VERDICT"]
        if vd:
            lines += [
                f"  {str(vd.get('rec', '?')).upper()}  conv {vd.get('conv', '?')}  {vd.get('horizon', '')}",
                f"  {vd.get('thesis', '')}",
            ]
            lines += [f"  + {p}" for p in vd.get("bull", [])]
            lines += [f"  - {p}" for p in vd.get("bear", [])]
            lines += [f"  ! {p}" for p in vd.get("risks", [])]
            lines.append(f"  flip: {vd.get('flip', '')}")
        else:
            lines.append(f"  unavailable ({res['news_engine']})")
        summary.delete("1.0", "end")
        summary.insert("end", "\n".join(lines))

        news.delete("1.0", "end")
        for n in res["news"]:
            sc = n.get("sentiment")
            score = f"[{sc:+.2f}]" if sc is not None else "[n/a]"
            news.insert("end", f"{score} {n['headline']} ({n['source']})\n")
        show_figures(chart_host, [res["figure"]])

    # ================= portfolio =================
    t2 = ttk.Frame(tabs)
    tabs.add(t2, text="Portfolio")
    c2 = ttk.Frame(t2)
    c2.pack(fill="x", pady=(8, 4))
    ptv = tk.StringVar(value="AAPL, MSFT, NVDA, GOOGL, AMZN, META")
    ttk.Label(c2, text="Tickers").pack(side="left", padx=8)
    ttk.Entry(c2, textvariable=ptv, width=70).pack(side="left")
    mv = tk.StringVar(value="HRP")
    ttk.Combobox(c2, textvariable=mv, state="readonly", width=10,
                 values=["HRP", "Markowitz"]).pack(side="left", padx=8)
    ttk.Button(c2, text="Optimize", command=lambda: run_portfolio()).pack(side="left", padx=8)
    body2 = ttk.Frame(t2)
    body2.pack(fill="both", expand=True)
    port = tk.Text(body2, width=36, font=("Courier", 10))
    port.pack(side="left", fill="y")
    p_outer, p_host = scroll_host(body2)
    p_outer.pack(side="left", fill="both", expand=True, padx=(8, 0))

    def render_portfolio(res: dict):
        port.delete("1.0", "end")
        port.insert("end", f"{res['method'].upper()} WEIGHTS\n")
        for k, v in res["weights"].items():
            if v > 0:
                port.insert("end", f"  {k:<8} {v:.2%}\n")
        er, vol, sh = res["performance"]
        port.insert("end", f"\n  ret {er:.2%}\n  vol {vol:.2%}\n  sharpe {sh:.2f}\n")
        show_figures(p_host, [res["figures"]["weights"], res["figures"]["correlation"]])

    # async plumbing: workers never touch Tk; results render on the main thread
    q = queue.Queue()

    def submit(work, render):
        def run():
            try:
                q.put(("ok", (render, work())))
            except Exception as e:
                q.put(("err", e))
        threading.Thread(target=run, daemon=True).start()

    def poll():
        try:
            while True:
                kind, payload = q.get_nowait()
                if kind == "ok":
                    render, result = payload
                    render(result)
                else:
                    messagebox.showerror("Error", str(payload))
        except queue.Empty:
            pass
        root.after(120, poll)

    # read Tk variables here, on the main thread
    def run_ticker():
        t = tv.get().strip().upper()
        if not t:
            return
        args = dict(ticker=t, benchmark=bv.get().strip().upper(), period=pv.get().strip(),
                    interval=iv.get(), pred_len=int(hv.get() or 21),
                    use_kronos=kv.get(), use_sentiment=sv.get(), use_verdict=sv.get())
        submit(lambda: analyze(**args), render_ticker)

    def run_portfolio():
        tickers = [t.strip().upper() for t in ptv.get().split(",") if t.strip()]
        if len(tickers) < 2:
            messagebox.showerror("Error", "Enter at least 2 tickers")
            return
        method = mv.get().lower()
        submit(lambda: optimize_portfolio(tickers, method=method), render_portfolio)

    root.after(120, poll)
    root.mainloop()


if __name__ == "__main__":
    run_ui()