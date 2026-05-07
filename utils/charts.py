"""
Módulo de visualizaciones con Plotly.
Incluye: precios, retornos, correlación, frontera eficiente, 
         drawdown, proyección de precio objetivo, evolución mensual de cartera, VaR.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


PALETTE = {
    "bg": "#0f1117",
    "card": "#1a1d27",
    "accent": "#00d4aa",
    "accent2": "#7c6af7",
    "accent3": "#f7b731",
    "accent4": "#fc5c65",
    "text": "#e8eaf0",
    "muted": "#6b7280",
    "green": "#26de81",
    "red": "#fc5c65",
    "grid": "rgba(255,255,255,0.05)",
}

PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="'IBM Plex Mono', monospace", color=PALETTE["text"], size=11),
    xaxis=dict(gridcolor=PALETTE["grid"], showline=False, tickfont=dict(size=10)),
    yaxis=dict(gridcolor=PALETTE["grid"], showline=False, tickfont=dict(size=10)),
    margin=dict(l=40, r=30, t=50, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor=PALETTE["grid"], borderwidth=1),
)

COLORS = [
    "#00d4aa", "#7c6af7", "#f7b731", "#fc5c65",
    "#26de81", "#45b7d1", "#fd9644", "#a55eea",
    "#eb3b5a", "#20bf6b", "#2d98da", "#f7b731",
]


def _apply_theme(fig, title=""):
    fig.update_layout(**PLOTLY_THEME, title=dict(text=title, font=dict(size=14, color=PALETTE["accent"])))
    return fig


# ── 1. Precio histórico normalizado ──────────────────────────────────────────

def chart_prices_normalized(prices: pd.DataFrame, title="Evolución de Precios (base 100)") -> go.Figure:
    fig = go.Figure()
    norm = prices / prices.iloc[0] * 100
    for i, col in enumerate(norm.columns):
        fig.add_trace(go.Scatter(
            x=norm.index, y=norm[col],
            name=col, mode="lines",
            line=dict(color=COLORS[i % len(COLORS)], width=2),
            hovertemplate=f"<b>{col}</b><br>%{{x|%d %b %Y}}<br>Índice: %{{y:.1f}}<extra></extra>"
        ))
    _apply_theme(fig, title)
    fig.update_layout(hovermode="x unified")
    return fig


# ── 2. Retornos acumulados ────────────────────────────────────────────────────

def chart_cumulative_returns(prices: pd.DataFrame) -> go.Figure:
    returns = prices.pct_change().dropna()
    cum = (1 + returns).cumprod() - 1
    fig = go.Figure()
    for i, col in enumerate(cum.columns):
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum[col] * 100,
            name=col, mode="lines",
            line=dict(color=COLORS[i % len(COLORS)], width=2),
            fill="tozeroy" if i == 0 else None,
            fillcolor="rgba(0,212,170,0.05)" if i == 0 else None,
        ))
    _apply_theme(fig, "Retornos Acumulados (%)")
    fig.update_layout(yaxis_ticksuffix="%", hovermode="x unified")
    return fig


# ── 3. Matriz de correlación ──────────────────────────────────────────────────

def chart_correlation_heatmap(prices: pd.DataFrame) -> go.Figure:
    returns = prices.pct_change().dropna()
    corr = returns.corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale=[
            [0, "#fc5c65"], [0.5, "#1a1d27"], [1, "#00d4aa"]
        ],
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=12),
        hovertemplate="<b>%{y} vs %{x}</b><br>Correlación: %{z:.3f}<extra></extra>",
        showscale=True,
        colorbar=dict(tickfont=dict(color=PALETTE["text"]))
    ))
    _apply_theme(fig, "Matriz de Correlación")
    return fig


# ── 4. Volatilidad rodante ────────────────────────────────────────────────────

def chart_rolling_volatility(prices: pd.DataFrame, window: int = 30) -> go.Figure:
    returns = prices.pct_change().dropna()
    roll_vol = returns.rolling(window).std() * np.sqrt(252) * 100
    fig = go.Figure()
    for i, col in enumerate(roll_vol.columns):
        fig.add_trace(go.Scatter(
            x=roll_vol.index, y=roll_vol[col],
            name=col, mode="lines",
            line=dict(color=COLORS[i % len(COLORS)], width=1.5),
        ))
    _apply_theme(fig, f"Volatilidad Rodante {window}d (Anualizada %)")
    fig.update_layout(yaxis_ticksuffix="%", hovermode="x unified")
    return fig


# ── 5. Frontera eficiente + Monte Carlo ───────────────────────────────────────

def chart_efficient_frontier(frontier_df: pd.DataFrame,
                              random_df: pd.DataFrame,
                              optimal_weights: np.ndarray,
                              mu: np.ndarray, sigma_mat: np.ndarray,
                              tickers: list,
                              selected_portfolio: dict) -> go.Figure:
    from models.optimizer import portfolio_performance

    fig = go.Figure()

    # Nube Monte Carlo
    fig.add_trace(go.Scatter(
        x=random_df["volatility"] * 100,
        y=random_df["return"] * 100,
        mode="markers",
        marker=dict(
            size=3, color=random_df["sharpe"],
            colorscale=[[0, "#1a1d27"], [0.5, "#7c6af7"], [1, "#00d4aa"]],
            opacity=0.5,
            colorbar=dict(title="Sharpe", tickfont=dict(color=PALETTE["text"]))
        ),
        name="Carteras Aleatorias",
        hovertemplate="Vol: %{x:.1f}%<br>Ret: %{y:.1f}%<extra></extra>"
    ))

    # Frontera eficiente
    ef = frontier_df.dropna()
    fig.add_trace(go.Scatter(
        x=ef["volatility"] * 100,
        y=ef["return"] * 100,
        mode="lines",
        line=dict(color=PALETTE["accent"], width=3),
        name="Frontera Eficiente",
    ))

    # Activos individuales
    for i, (t, m) in enumerate(zip(tickers, mu)):
        vol_i = float(np.sqrt(sigma_mat[i, i])) * 100
        fig.add_trace(go.Scatter(
            x=[vol_i], y=[m * 100],
            mode="markers+text",
            marker=dict(size=10, color=COLORS[i % len(COLORS)],
                        symbol="diamond", line=dict(width=1, color="white")),
            text=[t], textposition="top center",
            textfont=dict(size=9),
            name=t,
        ))

    # Cartera óptima seleccionada
    fig.add_trace(go.Scatter(
        x=[selected_portfolio["volatility"] * 100],
        y=[selected_portfolio["return"] * 100],
        mode="markers",
        marker=dict(size=18, color=PALETTE["accent3"],
                    symbol="star", line=dict(width=2, color="white")),
        name=f"★ Óptimo (Sharpe {selected_portfolio['sharpe']:.2f})",
    ))

    _apply_theme(fig, "Frontera Eficiente")
    fig.update_layout(
        xaxis_title="Volatilidad Anualizada (%)",
        yaxis_title="Retorno Esperado Anualizado (%)",
        xaxis_ticksuffix="%",
        yaxis_ticksuffix="%",
    )
    return fig


# ── 6. Pesos de cartera (dona + barras) ──────────────────────────────────────

def chart_portfolio_weights(weights: np.ndarray, tickers: list) -> go.Figure:
    fig = make_subplots(rows=1, cols=2,
                         specs=[[{"type": "domain"}, {"type": "bar"}]],
                         subplot_titles=["Distribución", "Pesos por Activo"])

    colors = [COLORS[i % len(COLORS)] for i in range(len(tickers))]
    w_pct = weights * 100

    fig.add_trace(go.Pie(
        labels=tickers, values=w_pct,
        hole=0.55,
        marker_colors=colors,
        textfont=dict(size=11),
        hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=tickers, y=w_pct,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in w_pct],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>%{y:.1f}%<extra></extra>",
    ), row=1, col=2)

    _apply_theme(fig, "Composición de la Cartera")
    fig.update_layout(showlegend=False)
    fig.update_yaxes(ticksuffix="%", row=1, col=2)
    return fig


# ── 7. Precio objetivo (estilo imagen adjunta) ───────────────────────────────

def chart_price_target(prices: pd.Series, projection: dict, ticker: str) -> go.Figure:
    fig = go.Figure()

    # Histórico
    fig.add_trace(go.Scatter(
        x=prices.index, y=prices.values,
        mode="lines", name="Histórico",
        line=dict(color=PALETTE["accent"], width=2),
    ))

    dates = projection["dates"]
    S0 = projection["S0"]

    # Rango de confianza (P25-P75)
    fig.add_trace(go.Scatter(
        x=list(dates) + list(reversed(list(dates))),
        y=list(projection["p75"]) + list(reversed(list(projection["p25"]))),
        fill="toself",
        fillcolor="rgba(0,212,170,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Rango P25-P75",
        hoverinfo="skip",
    ))

    # Rango extendido (P10-P90)
    fig.add_trace(go.Scatter(
        x=list(dates) + list(reversed(list(dates))),
        y=list(projection["p90"]) + list(reversed(list(projection["p10"]))),
        fill="toself",
        fillcolor="rgba(0,212,170,0.07)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Rango P10-P90",
        hoverinfo="skip",
    ))

    # Mediana (precio objetivo base)
    fig.add_trace(go.Scatter(
        x=dates, y=projection["p50"],
        mode="lines", name="Mediana (base)",
        line=dict(color=PALETTE["accent"], width=2, dash="dot"),
    ))

    # Anotaciones de precio objetivo
    targets = {
        "Alcista": (projection["target_bull"], PALETTE["green"]),
        "Base": (projection["target_base"], PALETTE["accent"]),
        "Bajista": (projection["target_bear"], PALETTE["red"]),
    }
    for label, (val, color) in targets.items():
        pct = (val / S0 - 1) * 100
        fig.add_annotation(
            x=dates[-1], y=val,
            text=f"{label}: ${val:.2f} ({pct:+.1f}%)",
            showarrow=True, arrowhead=2,
            arrowcolor=color, font=dict(color=color, size=10),
            bgcolor=PALETTE["card"],
        )

    # Línea vertical de separación histórico/proyección
    fig.add_vline(
        x=prices.index[-1], line_dash="dash",
        line_color=PALETTE["muted"], opacity=0.6,
        annotation_text="Hoy", annotation_font_color=PALETTE["muted"],
    )

    _apply_theme(fig, f"Precio Objetivo — {ticker} ({projection['horizon_months']}m horizonte)")
    fig.update_layout(
        yaxis_tickprefix="$",
        hovermode="x unified",
    )
    return fig


# ── 8. Evolución proyectada de cartera (mensual) ─────────────────────────────

def chart_portfolio_projection_monthly(prices: pd.DataFrame,
                                        weights: np.ndarray,
                                        horizon_months: int = 24,
                                        initial_capital: float = 10000,
                                        n_sims: int = 300) -> go.Figure:
    """
    Evolución proyectada mensual de la cartera (Monte Carlo GBM).
    Muestra historial real + proyección futura.
    """
    returns = np.log(prices / prices.shift(1)).dropna()
    port_returns = returns.values @ weights
    mu_p = port_returns.mean()
    sigma_p = port_returns.std()

    # Historial real de la cartera
    hist_value = initial_capital * (1 + (prices.pct_change().dropna() @ weights)).cumprod()
    S0 = hist_value.iloc[-1]

    # Simulaciones futuras
    n_steps = horizon_months * 21
    sims = np.zeros((n_sims, n_steps + 1))
    sims[:, 0] = S0
    dt = 1
    for i in range(n_sims):
        Z = np.random.standard_normal(n_steps)
        for t in range(1, n_steps + 1):
            sims[:, t] = sims[:, t - 1] * np.exp(
                (mu_p - 0.5 * sigma_p**2) * dt + sigma_p * np.sqrt(dt) * Z[t - 1]
            )

    last_date = prices.index[-1]
    future_dates = pd.date_range(last_date, periods=n_steps + 1, freq="B")

    p10 = np.percentile(sims, 10, axis=0)
    p25 = np.percentile(sims, 25, axis=0)
    p50 = np.percentile(sims, 50, axis=0)
    p75 = np.percentile(sims, 75, axis=0)
    p90 = np.percentile(sims, 90, axis=0)

    # Calcular retorno mensual promedio histórico
    monthly_hist = hist_value.resample("ME").last()

    fig = go.Figure()

    # Fondo sombreado bandas
    fig.add_trace(go.Scatter(
        x=list(future_dates) + list(reversed(list(future_dates))),
        y=list(p90) + list(reversed(list(p10))),
        fill="toself", fillcolor="rgba(124,106,247,0.08)",
        line=dict(color="rgba(0,0,0,0)"), name="Rango P10-P90",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=list(future_dates) + list(reversed(list(future_dates))),
        y=list(p75) + list(reversed(list(p25))),
        fill="toself", fillcolor="rgba(124,106,247,0.2)",
        line=dict(color="rgba(0,0,0,0)"), name="Rango P25-P75",
        hoverinfo="skip",
    ))

    # Historial real
    fig.add_trace(go.Scatter(
        x=hist_value.index, y=hist_value.values,
        mode="lines", name="Valor Real",
        line=dict(color=PALETTE["accent"], width=2.5),
    ))

    # Proyección mediana
    fig.add_trace(go.Scatter(
        x=future_dates, y=p50,
        mode="lines", name="Proyección Mediana",
        line=dict(color=PALETTE["accent2"], width=2, dash="dot"),
    ))

    # Proyección optimista
    fig.add_trace(go.Scatter(
        x=future_dates, y=p90,
        mode="lines", name="Escenario Alcista (P90)",
        line=dict(color=PALETTE["green"], width=1.5, dash="dash"),
    ))

    # Proyección pesimista
    fig.add_trace(go.Scatter(
        x=future_dates, y=p10,
        mode="lines", name="Escenario Bajista (P10)",
        line=dict(color=PALETTE["red"], width=1.5, dash="dash"),
    ))

    # Marcadores mensuales reales
    fig.add_trace(go.Scatter(
        x=monthly_hist.index, y=monthly_hist.values,
        mode="markers", name="Cierre Mensual",
        marker=dict(size=6, color=PALETTE["accent3"],
                    symbol="circle", line=dict(width=1, color="white")),
    ))

    # Separador
    fig.add_vline(
        x=prices.index[-1], line_dash="dash",
        line_color=PALETTE["muted"], opacity=0.6,
        annotation_text="Proyección →",
        annotation_font_color=PALETTE["accent"],
    )

    # Anotaciones finales
    final_ret = (p50[-1] / initial_capital - 1) * 100
    fig.add_annotation(
        x=future_dates[-1], y=p50[-1],
        text=f"${p50[-1]:,.0f}<br>({final_ret:+.1f}%)",
        showarrow=True, arrowhead=2,
        font=dict(color=PALETTE["accent2"], size=11),
        bgcolor=PALETTE["card"],
    )

    _apply_theme(fig, f"Evolución Proyectada de Cartera — Capital inicial ${initial_capital:,.0f}")
    fig.update_layout(
        yaxis_tickprefix="$",
        hovermode="x unified",
    )
    return fig


# ── 9. Drawdown ───────────────────────────────────────────────────────────────

def chart_drawdown(prices: pd.DataFrame, weights: np.ndarray, tickers: list) -> go.Figure:
    port_returns = prices.pct_change().dropna() @ weights
    cum = (1 + port_returns).cumprod()
    roll_max = cum.cummax()
    drawdown = (cum - roll_max) / roll_max * 100

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         row_heights=[0.6, 0.4], vertical_spacing=0.05)

    fig.add_trace(go.Scatter(
        x=cum.index, y=cum.values,
        mode="lines", name="Valor Cartera",
        line=dict(color=PALETTE["accent"], width=2),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values,
        mode="lines", name="Drawdown",
        fill="tozeroy",
        fillcolor="rgba(252,92,101,0.2)",
        line=dict(color=PALETTE["red"], width=1.5),
    ), row=2, col=1)

    _apply_theme(fig, "Historial de Cartera & Drawdown")
    fig.update_yaxes(title_text="Valor (base 1)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", ticksuffix="%", row=2, col=1)
    return fig


# ── 10. Distribución de retornos + VaR ───────────────────────────────────────

def chart_returns_distribution(prices: pd.DataFrame, weights: np.ndarray,
                                 var_95: float, var_99: float) -> go.Figure:
    port_returns = (prices.pct_change().dropna() @ weights) * 100

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=port_returns,
        nbinsx=80,
        name="Retornos Diarios",
        marker_color=PALETTE["accent2"],
        opacity=0.7,
        hovertemplate="Ret: %{x:.2f}%<br>Frec: %{y}<extra></extra>",
    ))

    fig.add_vline(x=var_95 * 100, line_dash="dash", line_color=PALETTE["accent3"],
                   annotation_text=f"VaR 95%: {var_95*100:.2f}%",
                   annotation_font_color=PALETTE["accent3"])
    fig.add_vline(x=var_99 * 100, line_dash="dash", line_color=PALETTE["red"],
                   annotation_text=f"VaR 99%: {var_99*100:.2f}%",
                   annotation_font_color=PALETTE["red"])

    _apply_theme(fig, "Distribución de Retornos Diarios")
    fig.update_layout(xaxis_ticksuffix="%", yaxis_title="Frecuencia")
    return fig


# ── 11. Retorno/Riesgo por activo (scatter) ───────────────────────────────────

def chart_risk_return_scatter(mu: np.ndarray, sigma_mat: np.ndarray,
                               tickers: list) -> go.Figure:
    vols = np.sqrt(np.diag(sigma_mat)) * 100
    rets = mu * 100
    sharpes = [(r - 4.5) / v for r, v in zip(rets, vols)]

    fig = go.Figure()
    for i, (t, v, r, s) in enumerate(zip(tickers, vols, rets, sharpes)):
        fig.add_trace(go.Scatter(
            x=[v], y=[r],
            mode="markers+text",
            marker=dict(
                size=max(12, abs(s) * 8),
                color=COLORS[i % len(COLORS)],
                opacity=0.85,
                line=dict(width=1.5, color="white"),
            ),
            text=[t], textposition="top center",
            name=t,
            hovertemplate=f"<b>{t}</b><br>Riesgo: {v:.1f}%<br>Retorno: {r:.1f}%<br>Sharpe: {s:.2f}<extra></extra>",
        ))

    _apply_theme(fig, "Riesgo vs Retorno por Activo")
    fig.update_layout(
        xaxis_title="Volatilidad Anualizada (%)",
        yaxis_title="Retorno Esperado (%)",
        xaxis_ticksuffix="%",
        yaxis_ticksuffix="%",
        showlegend=False,
    )
    return fig


# ── 12. Backtesting vs benchmark ─────────────────────────────────────────────

def chart_backtest(backtest_df: pd.DataFrame, benchmark_prices: pd.Series = None) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=backtest_df.index, y=backtest_df["portfolio"],
        mode="lines", name="Mi Cartera",
        line=dict(color=PALETTE["accent"], width=2.5),
        fill="tozeroy", fillcolor="rgba(0,212,170,0.05)",
    ))

    if benchmark_prices is not None:
        bench_norm = benchmark_prices / benchmark_prices.iloc[0]
        bench_aligned = bench_norm.reindex(backtest_df.index, method="ffill")
        fig.add_trace(go.Scatter(
            x=bench_aligned.index, y=bench_aligned,
            mode="lines", name="Benchmark (SPY)",
            line=dict(color=PALETTE["muted"], width=1.5, dash="dot"),
        ))

    _apply_theme(fig, "Backtesting — Valor de la Cartera (base 1)")
    fig.update_layout(hovermode="x unified")
    return fig


# ── 13. Análisis de analistas (precio objetivo) ───────────────────────────────

def chart_analyst_targets(ticker_info: dict) -> go.Figure:
    """Visualiza precio actual vs rango de analistas (estilo imagen adjunta)."""
    current = ticker_info.get("current_price")
    low = ticker_info.get("analyst_low")
    base = ticker_info.get("analyst_target")
    high = ticker_info.get("52w_high")
    ticker = ticker_info.get("ticker", "")

    if not all([current, low, base]):
        return None

    fig = go.Figure()

    # Rango de analistas
    fig.add_trace(go.Scatter(
        x=[low, low, high or base * 1.15, high or base * 1.15, low],
        y=[0.3, 0.7, 0.7, 0.3, 0.3],
        fill="toself",
        fillcolor="rgba(0,212,170,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Rango Analistas",
    ))

    # Línea precio actual
    fig.add_shape(type="line", x0=current, x1=current, y0=0.2, y1=0.8,
                   line=dict(color=PALETTE["accent"], width=3))

    # Marcador objetivo promedio
    fig.add_shape(type="line", x0=base, x1=base, y0=0.25, y1=0.75,
                   line=dict(color=PALETTE["accent3"], width=2, dash="dot"))

    pct_upside = (base / current - 1) * 100

    fig.add_annotation(x=current, y=0.85, text=f"Actual: ${current:.2f}",
                        font=dict(color=PALETTE["accent"], size=12), showarrow=False)
    fig.add_annotation(x=base, y=0.15, text=f"Objetivo: ${base:.2f} ({pct_upside:+.1f}%)",
                        font=dict(color=PALETTE["accent3"], size=12), showarrow=False)

    _apply_theme(fig, f"Precio Objetivo Analistas — {ticker}")
    fig.update_layout(
        xaxis_tickprefix="$",
        yaxis=dict(visible=False),
        showlegend=False,
        height=200,
    )
    return fig
