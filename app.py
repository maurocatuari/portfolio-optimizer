"""
Portfolio Optimizer Pro — Aplicación principal Streamlit.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Asegurarse que el path raíz esté en sys.path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from data.downloader import (
    load_prices_cached, get_ticker_info,
    load_watchlist, save_watchlist
)
from models.optimizer import (
    compute_returns, compute_expected_returns, compute_covariance,
    optimize_portfolio, compute_efficient_frontier, compute_random_portfolios,
    compute_var, backtest_portfolio, project_price_target, TRADING_DAYS
)
from utils.charts import (
    chart_prices_normalized, chart_cumulative_returns,
    chart_correlation_heatmap, chart_rolling_volatility,
    chart_efficient_frontier, chart_portfolio_weights,
    chart_price_target, chart_portfolio_projection_monthly,
    chart_drawdown, chart_returns_distribution,
    chart_risk_return_scatter, chart_backtest, chart_analyst_targets,
)

# ── Configuración de página ───────────────────────────────────────────────────

st.set_page_config(
    page_title="Portfolio Optimizer Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personalizado
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

:root {
    --bg: #0f1117;
    --card: #1a1d27;
    --accent: #00d4aa;
    --accent2: #7c6af7;
    --accent3: #f7b731;
    --red: #fc5c65;
    --text: #e8eaf0;
    --muted: #6b7280;
}

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.main { background-color: var(--bg) !important; }
section[data-testid="stSidebar"] { background-color: #13151f !important; }

.metric-card {
    background: linear-gradient(135deg, #1a1d27 0%, #1e2133 100%);
    border: 1px solid rgba(0,212,170,0.15);
    border-radius: 12px;
    padding: 18px 20px;
    margin: 6px 0;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: rgba(0,212,170,0.4); }
.metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.12em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 4px;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 24px;
    font-weight: 600;
    color: var(--accent);
}
.metric-value.negative { color: var(--red); }
.metric-value.neutral { color: var(--accent3); }

.section-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.18em;
    color: var(--muted);
    text-transform: uppercase;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    padding-bottom: 8px;
    margin: 24px 0 16px 0;
}

.ticker-badge {
    display: inline-block;
    background: rgba(0,212,170,0.12);
    border: 1px solid rgba(0,212,170,0.3);
    border-radius: 6px;
    padding: 3px 10px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: var(--accent);
    margin: 2px;
}

.warning-box {
    background: rgba(247,183,49,0.1);
    border: 1px solid rgba(247,183,49,0.3);
    border-radius: 8px;
    padding: 12px 16px;
    color: var(--accent3);
    font-size: 13px;
}

.stButton > button {
    background: linear-gradient(135deg, #00d4aa, #00b894) !important;
    color: #0f1117 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    letter-spacing: 0.05em !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

div[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    color: var(--accent) !important;
    font-size: 1.6rem !important;
}

[data-testid="stTab"] { font-family: 'IBM Plex Mono', monospace !important; }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="padding: 20px 0 10px 0; border-bottom: 1px solid rgba(0,212,170,0.15); margin-bottom: 24px;">
    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 10px; letter-spacing: 0.2em; color: #6b7280; text-transform: uppercase; margin-bottom: 4px;">
        SISTEMA DE ANÁLISIS ◆ v1.0
    </div>
    <div style="font-family: 'Space Grotesk', sans-serif; font-size: 32px; font-weight: 700; color: #e8eaf0;">
        Portfolio <span style="color: #00d4aa;">Optimizer</span> Pro
    </div>
    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #6b7280; margin-top: 4px;">
        Optimización cuantitativa de carteras · Frontera eficiente · Simulación Monte Carlo
    </div>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="section-title">⚙ Configuración</div>', unsafe_allow_html=True)

    # Activos
    st.markdown("**Activos**")
    watchlist = load_watchlist()
    default_tickers = [a["ticker"] for a in watchlist]

    ticker_input = st.text_area(
        "Tickers (uno por línea o separados por coma)",
        value="\n".join(default_tickers),
        height=120,
        help="Ej: SPY, QQQ, AGG, GLD, BTC-USD, AAPL"
    )
    tickers = [t.strip().upper() for t in ticker_input.replace(",", "\n").split("\n") if t.strip()]

    st.markdown("---")
    st.markdown("**Período Histórico**")
    col_s, col_e = st.columns(2)
    with col_s:
        start_date = st.date_input("Desde", datetime.now() - timedelta(days=365*3))
    with col_e:
        end_date = st.date_input("Hasta", datetime.now())

    interval = st.selectbox("Frecuencia", ["1d", "1wk", "1mo"], index=0)

    st.markdown("---")
    st.markdown("**Optimización**")
    objective = st.selectbox("Objetivo", [
        "Máximo Sharpe Ratio",
        "Mínima Volatilidad",
        "Máximo Retorno",
        "Retorno Objetivo",
        "Volatilidad Objetivo",
    ])

    objective_map = {
        "Máximo Sharpe Ratio": "sharpe",
        "Mínima Volatilidad": "min_vol",
        "Máximo Retorno": "max_return",
        "Retorno Objetivo": "target_return",
        "Volatilidad Objetivo": "target_vol",
    }

    target_val = None
    if objective == "Retorno Objetivo":
        target_val = st.slider("Retorno anual objetivo (%)", 1, 50, 15) / 100
    elif objective == "Volatilidad Objetivo":
        target_val = st.slider("Volatilidad anual objetivo (%)", 5, 50, 20) / 100

    allow_short = st.checkbox("Permitir ventas en corto", value=False)
    max_weight_pct = st.slider("Peso máximo por activo (%)", 10, 100, 60)
    max_weight = max_weight_pct / 100

    st.markdown("---")
    st.markdown("**Modelo de Retornos**")
    return_model = st.selectbox("Método", ["historical", "ema", "capm"],
                                  format_func=lambda x: {
                                      "historical": "Media Histórica",
                                      "ema": "Media Exponencial (EMA)",
                                      "capm": "CAPM"
                                  }[x])

    cov_method = st.selectbox("Covarianza", ["sample", "ledoit_wolf"],
                               format_func=lambda x: {
                                   "sample": "Muestral",
                                   "ledoit_wolf": "Ledoit-Wolf (shrinkage)"
                               }[x])

    st.markdown("---")
    st.markdown("**Proyecciones**")
    initial_capital = st.number_input("Capital inicial ($)", 1000, 1_000_000, 10_000, step=1000)
    horizon_months = st.slider("Horizonte proyección (meses)", 3, 60, 24)
    projection_ticker = st.selectbox("Ticker para precio objetivo", tickers if tickers else ["SPY"])

    st.markdown("---")
    st.markdown("**Backtesting**")
    rebalance_freq_map = {
        "Buy & Hold": None,
        "Mensual": "ME",
        "Trimestral": "QE",
        "Anual": "YE",
    }
    rebalance_label = st.selectbox("Frecuencia de rebalanceo", list(rebalance_freq_map.keys()))
    rebalance_freq = rebalance_freq_map[rebalance_label]

    force_refresh = st.checkbox("Forzar descarga (ignorar caché)", value=False)

    run_btn = st.button("🚀 Calcular Cartera", use_container_width=True)


# ── Lógica principal ──────────────────────────────────────────────────────────

if run_btn or "portfolio_result" in st.session_state:

    if run_btn:
        if len(tickers) < 2:
            st.error("⚠ Necesitás al menos 2 activos para optimizar una cartera.")
            st.stop()

        with st.spinner("📡 Descargando datos financieros..."):
            prices, failed, from_cache = load_prices_cached(
                tickers,
                str(start_date), str(end_date),
                interval=interval,
                force_refresh=force_refresh
            )

        if failed:
            st.markdown(f'<div class="warning-box">⚠ No se pudieron descargar: {", ".join(failed)}</div>',
                        unsafe_allow_html=True)

        if prices.empty or len(prices.columns) < 2:
            st.error("No hay suficientes datos para continuar.")
            st.stop()

        # Calcular retornos y matrices
        with st.spinner("⚙ Optimizando cartera..."):
            returns = compute_returns(prices, method="log")
            mu = compute_expected_returns(returns, method=return_model)
            sigma = compute_covariance(returns, method=cov_method)
            n = len(prices.columns)
            tickers_ok = list(prices.columns)

            # Optimizar
            opt_kwargs = {"objective": objective_map[objective],
                          "allow_short": allow_short, "max_weight": max_weight}
            if target_val is not None:
                if "retorno" in objective.lower():
                    opt_kwargs["target_return"] = target_val
                else:
                    opt_kwargs["target_vol"] = target_val

            result = optimize_portfolio(mu, sigma, n, **opt_kwargs)
            weights = result["weights"]

            # Frontera eficiente + Monte Carlo
            frontier = compute_efficient_frontier(mu, sigma, n, allow_short=allow_short, max_weight=max_weight)
            random_p = compute_random_portfolios(mu, sigma, n, n_portfolios=2000)

            # VaR
            var_data = compute_var(returns, weights, confidence=0.95)
            var_data_99 = compute_var(returns, weights, confidence=0.99)

            # Proyección de precio objetivo
            if projection_ticker in prices.columns:
                projection = project_price_target(prices[projection_ticker], horizon_months=horizon_months)
            else:
                projection = None

            # Proyección de cartera
            port_projection = chart_portfolio_projection_monthly(
                prices, weights,
                horizon_months=horizon_months,
                initial_capital=initial_capital,
            )

            # Backtesting
            backtest_df, bt_metrics = backtest_portfolio(prices, weights, rebalance_freq=rebalance_freq)

        # Info de tickers
        with st.spinner("📋 Obteniendo info de activos..."):
            ticker_infos = {}
            for t in tickers_ok:
                try:
                    ticker_infos[t] = get_ticker_info(t)
                except:
                    ticker_infos[t] = {"ticker": t, "name": t}

        # Guardar en session
        st.session_state["portfolio_result"] = {
            "prices": prices, "returns": returns,
            "mu": mu, "sigma": sigma, "weights": weights,
            "tickers": tickers_ok, "result": result,
            "frontier": frontier, "random_p": random_p,
            "var_data": var_data, "var_data_99": var_data_99,
            "projection": projection, "port_projection": port_projection,
            "backtest_df": backtest_df, "bt_metrics": bt_metrics,
            "ticker_infos": ticker_infos, "from_cache": from_cache,
            "initial_capital": initial_capital,
        }

    # ── Recuperar datos ───────────────────────────────────────────────────────
    d = st.session_state["portfolio_result"]
    prices = d["prices"]
    returns = d["returns"]
    mu = d["mu"]
    sigma = d["sigma"]
    weights = d["weights"]
    tickers_ok = d["tickers"]
    result = d["result"]
    frontier = d["frontier"]
    random_p = d["random_p"]
    var_data = d["var_data"]
    var_data_99 = d["var_data_99"]
    projection = d["projection"]
    port_projection = d["port_projection"]
    backtest_df = d["backtest_df"]
    bt_metrics = d["bt_metrics"]
    ticker_infos = d["ticker_infos"]
    initial_capital = d["initial_capital"]

    if d["from_cache"] and not run_btn:
        st.info("📦 Usando datos cacheados. Marcá 'Forzar descarga' para actualizar.")

    # ── KPIs principales ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title">◆ Métricas de la Cartera Óptima</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (c1, "Retorno Esperado", f"{result['return']*100:.2f}%", result['return'] > 0),
        (c2, "Volatilidad Anual", f"{result['volatility']*100:.2f}%", True),
        (c3, "Sharpe Ratio", f"{result['sharpe']:.3f}", result['sharpe'] > 1),
        (c4, "VaR Diario 95%", f"{var_data['var_daily']*100:.2f}%", False),
        (c5, "Drawdown Máx.", f"{(backtest_df['drawdown'].min()*100):.1f}%", False),
    ]
    for col, label, val, positive in kpis:
        color_class = "metric-value" if positive else "metric-value negative"
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="{color_class}">{val}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Tabla de pesos ────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">◆ Composición de la Cartera</div>', unsafe_allow_html=True)

    weight_df = pd.DataFrame({
        "Ticker": tickers_ok,
        "Nombre": [ticker_infos.get(t, {}).get("name", t) for t in tickers_ok],
        "Peso (%)": [f"{w*100:.2f}%" for w in weights],
        "Retorno Esp. (%)": [f"{m*100:.2f}%" for m in mu],
        "Volatilidad (%)": [f"{np.sqrt(sigma[i,i])*100:.2f}%" for i in range(len(tickers_ok))],
        "Beta": [ticker_infos.get(t, {}).get("beta", "N/A") for t in tickers_ok],
        "Sector": [ticker_infos.get(t, {}).get("sector", "N/A") for t in tickers_ok],
    })

    st.dataframe(
        weight_df,
        use_container_width=True,
        hide_index=True,
    )

    # ── Tabs de gráficos ──────────────────────────────────────────────────────
    tab_labels = [
        "📈 Precios", "⚡ Frontera", "🥧 Cartera",
        "🎯 Proyecciones", "📉 Riesgo", "🔄 Backtest", "🏦 Analistas"
    ]
    tabs = st.tabs(tab_labels)

    # ── Tab 1: Precios ────────────────────────────────────────────────────────
    with tabs[0]:
        st.plotly_chart(chart_prices_normalized(prices), use_container_width=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(chart_cumulative_returns(prices), use_container_width=True)
        with col_b:
            st.plotly_chart(chart_rolling_volatility(prices), use_container_width=True)
        st.plotly_chart(chart_correlation_heatmap(prices), use_container_width=True)

    # ── Tab 2: Frontera eficiente ──────────────────────────────────────────────
    with tabs[1]:
        st.plotly_chart(
            chart_efficient_frontier(frontier, random_p, weights, mu, sigma, tickers_ok, result),
            use_container_width=True
        )
        st.plotly_chart(chart_risk_return_scatter(mu, sigma, tickers_ok), use_container_width=True)

    # ── Tab 3: Cartera ─────────────────────────────────────────────────────────
    with tabs[2]:
        st.plotly_chart(chart_portfolio_weights(weights, tickers_ok), use_container_width=True)

    # ── Tab 4: Proyecciones ────────────────────────────────────────────────────
    with tabs[3]:
        st.plotly_chart(port_projection, use_container_width=True)

        if projection:
            st.markdown("---")
            st.markdown(f"**Precio Objetivo — {projection_ticker}**")

            col_t1, col_t2, col_t3 = st.columns(3)
            S0 = projection["S0"]
            for col, (label, key, color) in zip(
                [col_t1, col_t2, col_t3],
                [("Bajista (P25)", "target_bear", "#fc5c65"),
                 ("Base (P50)", "target_base", "#00d4aa"),
                 ("Alcista (P75)", "target_bull", "#26de81")]
            ):
                val = projection[key]
                pct = (val / S0 - 1) * 100
                with col:
                    st.markdown(f"""
                    <div class="metric-card" style="border-color: {color}33;">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value" style="color: {color};">${val:.2f}</div>
                        <div style="font-family: 'IBM Plex Mono'; font-size: 12px; color: {color}; opacity: 0.8;">{pct:+.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.plotly_chart(
                chart_price_target(prices[projection_ticker], projection, projection_ticker),
                use_container_width=True
            )

    # ── Tab 5: Riesgo ──────────────────────────────────────────────────────────
    with tabs[4]:
        st.plotly_chart(chart_drawdown(prices, weights, tickers_ok), use_container_width=True)
        st.plotly_chart(
            chart_returns_distribution(prices, weights,
                                        var_data["var_daily"],
                                        var_data_99["var_daily"]),
            use_container_width=True
        )

        col_v1, col_v2, col_v3, col_v4 = st.columns(4)
        with col_v1:
            st.metric("VaR 95% Diario", f"{var_data['var_daily']*100:.2f}%")
        with col_v2:
            st.metric("CVaR 95% Diario", f"{var_data['cvar_daily']*100:.2f}%")
        with col_v3:
            st.metric("VaR 99% Diario", f"{var_data_99['var_daily']*100:.2f}%")
        with col_v4:
            st.metric("Drawdown Máx.", f"{backtest_df['drawdown'].min()*100:.1f}%")

    # ── Tab 6: Backtest ────────────────────────────────────────────────────────
    with tabs[5]:
        bench_prices = None
        if "SPY" in prices.columns:
            bench_prices = prices["SPY"]

        st.plotly_chart(chart_backtest(backtest_df, bench_prices), use_container_width=True)

        bt_col1, bt_col2, bt_col3, bt_col4 = st.columns(4)
        with bt_col1:
            st.metric("Retorno Total", f"{bt_metrics['total_return']*100:.1f}%")
        with bt_col2:
            st.metric("Retorno Anualizado", f"{bt_metrics['ann_return']*100:.1f}%")
        with bt_col3:
            st.metric("Volatilidad Anual", f"{bt_metrics['ann_vol']*100:.1f}%")
        with bt_col4:
            st.metric("Sharpe Backtest", f"{bt_metrics['sharpe']:.3f}")

    # ── Tab 7: Analistas ───────────────────────────────────────────────────────
    with tabs[6]:
        for t in tickers_ok:
            info = ticker_infos.get(t, {})
            if info.get("analyst_target"):
                st.markdown(f"**{t}** — {info.get('name', t)}")
                fig_a = chart_analyst_targets(info)
                if fig_a:
                    st.plotly_chart(fig_a, use_container_width=True)

                a_col1, a_col2, a_col3, a_col4 = st.columns(4)
                with a_col1:
                    st.metric("Precio Actual", f"${info.get('current_price', 'N/A')}")
                with a_col2:
                    st.metric("Objetivo Analistas", f"${info.get('analyst_target', 'N/A')}")
                with a_col3:
                    st.metric("Mín. 52s", f"${info.get('52w_low', 'N/A')}")
                with a_col4:
                    st.metric("Máx. 52s", f"${info.get('52w_high', 'N/A')}")

                if info.get("description"):
                    with st.expander("Descripción"):
                        st.write(info["description"])
                st.markdown("---")

else:
    # ── Estado inicial ────────────────────────────────────────────────────────
    st.markdown("""
    <div style="
        text-align: center;
        padding: 80px 40px;
        background: linear-gradient(135deg, #1a1d27 0%, #13151f 100%);
        border: 1px solid rgba(0,212,170,0.1);
        border-radius: 16px;
        margin-top: 20px;
    ">
        <div style="font-size: 48px; margin-bottom: 16px;">📊</div>
        <div style="font-family: 'Space Grotesk'; font-size: 22px; font-weight: 600; color: #e8eaf0; margin-bottom: 8px;">
            Configurá tu cartera en el panel izquierdo
        </div>
        <div style="font-family: 'IBM Plex Mono'; font-size: 13px; color: #6b7280; max-width: 500px; margin: 0 auto; line-height: 1.8;">
            1. Ingresá los tickers de los activos<br>
            2. Seleccioná el período histórico<br>
            3. Elegí el objetivo de optimización<br>
            4. Presioná <strong style="color:#00d4aa">Calcular Cartera</strong>
        </div>
        <div style="margin-top: 32px; font-family: 'IBM Plex Mono'; font-size: 11px; color: #374151;">
            Ejemplo: SPY · QQQ · AGG · GLD · BTC-USD · AAPL · MSFT
        </div>
    </div>
    """, unsafe_allow_html=True)
