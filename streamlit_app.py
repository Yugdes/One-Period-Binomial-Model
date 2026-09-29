from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# -----------------------------
# Model Core (your binomial logic)
# -----------------------------
@dataclass
class BinomialResult:
    spot: float
    strike: float
    maturity_years: float
    volatility: float
    r: float
    u: float
    d: float
    su: float
    sd: float
    payoff_up: float
    payoff_down: float
    delta: float
    bond: float
    risk_neutral_p: float
    model_price: float


def extract_close_series_from_download(df: pd.DataFrame, symbol: str) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)

    if isinstance(df.columns, pd.MultiIndex):
        if ("Close", symbol) in df.columns:
            s = df[("Close", symbol)]
        elif "Close" in df.columns.get_level_values(0):
            s = df.xs("Close", axis=1, level=0).iloc[:, 0]
        else:
            raise ValueError("Could not find Close column in intraday data.")
    else:
        if "Close" not in df.columns:
            raise ValueError("Could not find Close column in intraday data.")
        s = df["Close"]

    return pd.to_numeric(s, errors="coerce").dropna()


def realized_volatility_daily(close_prices: pd.Series, window: int = 60) -> float:
    log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
    if log_returns.empty:
        raise ValueError("Not enough history to compute daily volatility.")
    if window > 1:
        log_returns = log_returns.tail(window)
    sigma_daily = float(log_returns.std(ddof=1))
    sigma_annual = sigma_daily * np.sqrt(252.0)
    if sigma_annual <= 0:
        raise ValueError("Computed daily volatility is non-positive.")
    return sigma_annual


def realized_volatility_last_minutes(
    symbol: str,
    interval: str = "1m",
    period: str = "5d",
    window_minutes: int = 60,
) -> Tuple[float, float, pd.Series]:
    raw = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    close = extract_close_series_from_download(raw, symbol)
    close_tail = close.tail(window_minutes)

    if len(close_tail) < max(10, window_minutes // 3):
        raise ValueError(
            f"Not enough intraday bars for last {window_minutes} minutes. Found {len(close_tail)} bars."
        )

    r_1m = np.log(close_tail / close_tail.shift(1)).dropna()
    if len(r_1m) < 2:
        raise ValueError("Not enough 1-minute returns for volatility.")

    sigma_1m = float(r_1m.std(ddof=1))
    sigma_1h = sigma_1m * np.sqrt(60.0)

    minutes_per_year = 252.0 * 6.5 * 60.0
    sigma_annual = sigma_1m * np.sqrt(minutes_per_year)

    if sigma_annual <= 0:
        raise ValueError("Computed intraday volatility is non-positive.")

    return float(sigma_annual), float(sigma_1h), close_tail


def one_period_binomial(
    spot: float,
    strike: float,
    r: float,
    maturity_years: float,
    sigma_annual: float,
    option_type: str,
) -> BinomialResult:
    dt = maturity_years
    u = float(np.exp(sigma_annual * np.sqrt(dt)))
    d = float(np.exp(-sigma_annual * np.sqrt(dt)))
    growth = float(np.exp(r * dt))

    if not (d < growth < u):
        raise ValueError("No-arbitrage condition failed: need d < exp(rT) < u.")

    su = spot * u
    sd = spot * d

    if option_type.lower() == "call":
        payoff_up = max(su - strike, 0.0)
        payoff_down = max(sd - strike, 0.0)
    elif option_type.lower() == "put":
        payoff_up = max(strike - su, 0.0)
        payoff_down = max(strike - sd, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'.")

    delta = (payoff_up - payoff_down) / (su - sd)
    bond = np.exp(-r * dt) * (payoff_up - delta * su)

    p = (growth - d) / (u - d)
    model_price_rn = np.exp(-r * dt) * (p * payoff_up + (1.0 - p) * payoff_down)
    model_price_rep = delta * spot + bond

    if abs(model_price_rn - model_price_rep) > 1e-8:
        raise RuntimeError("Replicating and risk-neutral prices do not match.")

    return BinomialResult(
        spot=spot,
        strike=strike,
        maturity_years=dt,
        volatility=sigma_annual,
        r=r,
        u=u,
        d=d,
        su=su,
        sd=sd,
        payoff_up=payoff_up,
        payoff_down=payoff_down,
        delta=delta,
        bond=bond,
        risk_neutral_p=p,
        model_price=model_price_rep,
    )


def investment_signal(model_price: float, market_price: float, threshold: float) -> str:
    relative_gap = (model_price - market_price) / market_price
    if relative_gap > threshold:
        return "BUY option (model > market: potentially undervalued)"
    if relative_gap < -threshold:
        return "SELL/WRITE option (model < market: potentially overvalued)"
    return "HOLD/NO TRADE (difference within threshold band)"


def get_live_spot_price(ticker: yf.Ticker, fallback_close: float) -> float:
    try:
        info = ticker.fast_info
        if hasattr(info, "get"):
            for key in ("lastPrice", "last_price", "regularMarketPrice", "regular_market_price"):
                value = info.get(key, None)
                if value is not None and float(value) > 0:
                    return float(value)
    except Exception:
        pass
    return float(fallback_close)


def extract_market_option_price(chain_df: pd.DataFrame, strike: float) -> Tuple[float, float, float, float]:
    row = chain_df.loc[(chain_df["strike"] - strike).abs().idxmin()]
    bid = float(row.get("bid", 0.0) or 0.0)
    ask = float(row.get("ask", 0.0) or 0.0)
    last = float(row.get("lastPrice", np.nan))
    iv = float(row.get("impliedVolatility", np.nan))

    if bid > 0 and ask > 0:
        mid = (bid + ask) / 2.0
        return mid, bid, ask, iv
    if np.isfinite(last) and last > 0:
        return last, bid, ask, iv
    raise ValueError("Could not infer market option price from bid/ask/last.")


# -----------------------------
# Data Access
# -----------------------------
@st.cache_data(ttl=90, show_spinner=False)
def fetch_daily_history(symbol: str, period: str) -> pd.DataFrame:
    return yf.Ticker(symbol).history(period=period, auto_adjust=True)


@st.cache_data(ttl=90, show_spinner=False)
def fetch_expiries(symbol: str) -> list[str]:
    return list(yf.Ticker(symbol).options)


@st.cache_data(ttl=60, show_spinner=False)
def fetch_option_chain(symbol: str, expiry: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    chain = yf.Ticker(symbol).option_chain(expiry)
    return chain.calls, chain.puts


def signal_box(signal: str) -> None:
    if signal.startswith("BUY"):
        st.success(signal)
    elif signal.startswith("SELL"):
        st.error(signal)
    else:
        st.info(signal)


def main() -> None:
    st.set_page_config(page_title="Options Dashboard • Binomial Model", layout="wide")
    st.title("Options Dashboard (One-Period Binomial)")
    st.caption("Live Yahoo data + your binomial model for valuation and trade signal")

    popular_tickers = [
        "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL", "NFLX",
        "AMD", "INTC", "SPY", "QQQ", "RELIANCE.NS", "TCS.NS", "INFY.NS"
    ]

    with st.sidebar:
        st.header("Instrument")
        ticker_pick = st.selectbox("Stock (searchable)", options=popular_tickers, index=0)
        ticker_manual = st.text_input("Or type any ticker", value=ticker_pick)
        symbol = ticker_manual.strip().upper()

        st.header("Contract")
        option_type = st.selectbox("Option type", options=["call", "put"], index=0)

        st.header("Model Inputs")
        risk_free_rate = st.number_input("Risk-free rate (decimal)", min_value=0.0, max_value=1.0, value=0.05, step=0.005, format="%.4f")
        threshold = st.number_input("Signal threshold", min_value=0.0, max_value=1.0, value=0.05, step=0.01, format="%.4f")
        contracts = st.number_input("Contracts", min_value=1, value=1, step=1)
        contract_size = st.number_input("Contract size", min_value=1, value=100, step=1)

        st.header("Volatility")
        vol_mode = st.selectbox("Volatility mode", options=["daily", "intraday60"], index=0)
        history_period = st.selectbox("Daily history period", options=["6mo", "1y", "2y", "5y"], index=1)
        vol_window = st.slider("Daily vol window", min_value=20, max_value=252, value=60, step=1)
        intraday_interval = st.selectbox("Intraday interval", options=["1m", "2m", "5m", "15m"], index=0)
        intraday_period = st.selectbox("Intraday lookback period", options=["1d", "5d", "1mo"], index=1)
        intraday_window_minutes = st.slider("Intraday minutes window", min_value=30, max_value=390, value=60, step=5)

        st.header("Horizon")
        one_hour_step = st.checkbox("Use 1-hour step (ignore expiry horizon)", value=False)

    if not symbol:
        st.warning("Please enter a ticker.")
        st.stop()

    try:
        ticker = yf.Ticker(symbol)
        hist = fetch_daily_history(symbol, history_period)
        if hist.empty:
            st.error(f"No historical data found for {symbol}.")
            st.stop()

        close_daily = hist["Close"].dropna()
        last_close = float(close_daily.iloc[-1])
        spot = get_live_spot_price(ticker, fallback_close=last_close)

        expiries = fetch_expiries(symbol)
        if not expiries:
            st.error(f"No option expiries found for {symbol}.")
            st.stop()

        with st.sidebar:
            expiry = st.selectbox("Option expiry", options=expiries, index=0, help="Only listed expiry dates are available.")

        calls_df, puts_df = fetch_option_chain(symbol, expiry)
        chain_df = calls_df if option_type == "call" else puts_df
        if chain_df.empty:
            st.error(f"No {option_type} data for {symbol} at expiry {expiry}.")
            st.stop()

        strikes = sorted(chain_df["strike"].dropna().astype(float).unique().tolist())
        default_strike = min(strikes, key=lambda x: abs(x - spot))

        with st.sidebar:
            strike = st.selectbox("Strike", options=strikes, index=strikes.index(default_strike))

        if vol_mode == "intraday60":
            sigma, sigma_1h, close_for_chart = realized_volatility_last_minutes(
                symbol=symbol,
                interval=intraday_interval,
                period=intraday_period,
                window_minutes=intraday_window_minutes,
            )
            vol_source = f"intraday {intraday_interval}, last {intraday_window_minutes} mins"
        else:
            sigma = realized_volatility_daily(close_daily, window=vol_window)
            sigma_1h = None
            close_for_chart = close_daily
            vol_source = f"daily close returns, window={vol_window}"

        expiry_dt = datetime.strptime(expiry, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        t_to_expiry = (expiry_dt - now).total_seconds() / (365.0 * 24.0 * 3600.0)
        if t_to_expiry <= 0:
            st.error(f"Selected expiry {expiry} is not in the future.")
            st.stop()

        maturity_years = (1.0 / (252.0 * 6.5)) if one_hour_step else t_to_expiry

        market_price, bid, ask, market_iv = extract_market_option_price(chain_df, strike)
        result = one_period_binomial(
            spot=spot,
            strike=strike,
            r=risk_free_rate,
            maturity_years=maturity_years,
            sigma_annual=sigma,
            option_type=option_type,
        )

        signal = investment_signal(result.model_price, market_price, threshold)
        rel_gap = (result.model_price - market_price) / market_price

        mult = int(contracts) * int(contract_size)
        shares_units = result.delta * mult
        cash_units_now = result.bond * mult
        cash_units_step = cash_units_now * np.exp(result.r * result.maturity_years)
        rep_now = result.model_price * mult
        rep_up = shares_units * result.su + cash_units_step
        rep_down = shares_units * result.sd + cash_units_step
        model_total = result.model_price * mult
        market_total = market_price * mult

        # Top metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Spot", f"{result.spot:.2f}")
        c2.metric("Strike", f"{result.strike:.2f}")
        c3.metric("Model Price", f"{result.model_price:.4f}")
        c4.metric("Market Price", f"{market_price:.4f}")
        c5.metric("Mispricing", f"{rel_gap:.2%}")
        signal_box(signal)

        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Model Internals", "Chain Snapshot", "Replication"])

        with tab1:
            left, right = st.columns([2, 1])

            with left:
                fig_price = go.Figure()
                if {"Open", "High", "Low", "Close"}.issubset(set(hist.columns)):
                    hist_tail = hist.tail(120)
                    fig_price.add_trace(
                        go.Candlestick(
                            x=hist_tail.index,
                            open=hist_tail["Open"],
                            high=hist_tail["High"],
                            low=hist_tail["Low"],
                            close=hist_tail["Close"],
                            name="OHLC",
                        )
                    )
                fig_price.add_trace(
                    go.Scatter(
                        x=close_for_chart.index,
                        y=close_for_chart.values,
                        mode="lines",
                        name="Close",
                        line=dict(width=2),
                    )
                )
                fig_price.add_hline(y=result.strike, line_dash="dash", line_color="orange", annotation_text="Strike")
                fig_price.add_hline(y=result.su, line_dash="dot", line_color="green", annotation_text="Su")
                fig_price.add_hline(y=result.sd, line_dash="dot", line_color="red", annotation_text="Sd")
                fig_price.update_layout(
                    title=f"{symbol} Price Chart",
                    height=500,
                    xaxis_title="Time",
                    yaxis_title="Price",
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig_price, width="stretch")

            with right:
                fig_bar = go.Figure(
                    data=[
                        go.Bar(
                            x=["Model Total", "Market Total", "Rep Up", "Rep Down"],
                            y=[model_total, market_total, rep_up, rep_down],
                            marker_color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
                        )
                    ]
                )
                fig_bar.update_layout(
                    title="Portfolio Value View",
                    height=500,
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig_bar, width="stretch")

        with tab2:
            m1, m2 = st.columns(2)
            with m1:
                st.markdown(
                    "\n".join(
                        [
                            f"- Symbol: `{symbol}`",
                            f"- Option: `{option_type.upper()}`",
                            f"- Expiry: `{expiry}`",
                            f"- Vol source: `{vol_source}`",
                            f"- Sigma (annualized): `{result.volatility:.4%}`",
                            f"- Maturity T: `{result.maturity_years:.8f}` years",
                            f"- Risk-free r: `{result.r:.4%}`",
                            f"- u, d: `{result.u:.6f}`, `{result.d:.6f}`",
                            f"- Su, Sd: `{result.su:.4f}`, `{result.sd:.4f}`",
                        ]
                    )
                )
                if sigma_1h is not None:
                    st.markdown(f"- Sigma over 1 hour: `{sigma_1h:.6f}`")
            with m2:
                st.markdown(
                    "\n".join(
                        [
                            f"- Payoff Up/Down: `{result.payoff_up:.4f}` / `{result.payoff_down:.4f}`",
                            f"- Risk-neutral p: `{result.risk_neutral_p:.6f}`",
                            f"- Delta: `{result.delta:.6f}`",
                            f"- Bond B: `{result.bond:.6f}`",
                            f"- Model option price: `{result.model_price:.4f}`",
                            f"- Market option price: `{market_price:.4f}`",
                            f"- Decision threshold: `{threshold:.2%}`",
                        ]
                    )
                )

        with tab3:
            nearest_row = chain_df.loc[(chain_df["strike"] - strike).abs().idxmin()].copy()
            show_cols = [
                "strike", "lastPrice", "bid", "ask", "volume", "openInterest",
                "impliedVolatility", "inTheMoney"
            ]
            show_cols = [c for c in show_cols if c in chain_df.columns]
            st.dataframe(chain_df[show_cols].sort_values("strike").reset_index(drop=True), width="stretch")
            st.write("Selected strike row:")
            st.dataframe(pd.DataFrame([nearest_row[show_cols].to_dict()]), width="stretch")
            st.caption(f"Bid={bid:.4f} | Ask={ask:.4f} | Market IV={market_iv if np.isfinite(market_iv) else np.nan}")

        with tab4:
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Contracts x Size", f"{mult}")
            r2.metric("Replicating Shares Qty", f"{shares_units:.4f}")
            r3.metric("Replicating Cash Today", f"{cash_units_now:.4f}")
            r4.metric("Replicating Value Now", f"{rep_now:.4f}")

            st.markdown(
                "\n".join(
                    [
                        f"- Replicating portfolio in Up state: `{rep_up:.4f}`",
                        f"- Replicating portfolio in Down state: `{rep_down:.4f}`",
                        f"- Cash grown to step: `{cash_units_step:.4f}`",
                        f"- Model total value: `{model_total:.4f}`",
                        f"- Market total value: `{market_total:.4f}`",
                    ]
                )
            )

        st.caption(
            f"Last updated (UTC): {now.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"Data source: Yahoo Finance | Dashboard run mode: {'1-hour step' if one_hour_step else 'expiry horizon'}"
        )

    except Exception as exc:
        st.error(f"Computation failed: {exc}")


if __name__ == "__main__":
    main()
