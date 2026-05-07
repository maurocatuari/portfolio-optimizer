"""
Módulo de optimización de cartera.
Implementa: Markowitz (frontera eficiente), Máximo Sharpe, Mínima varianza,
Black-Litterman simplificado, métricas de riesgo avanzadas.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

TRADING_DAYS = 252
RISK_FREE_RATE_ANNUAL = 0.045  # 4.5% anual (ajustar según contexto)


# ── Retornos y covarianza ─────────────────────────────────────────────────────

def compute_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """Calcula retornos diarios. method: 'log' o 'simple'"""
    if method == "log":
        return np.log(prices / prices.shift(1)).dropna()
    return prices.pct_change().dropna()


def compute_expected_returns(returns: pd.DataFrame, method: str = "historical") -> np.ndarray:
    """
    Calcula retorno esperado anualizado.
    method: 'historical' | 'capm' | 'ema' (exponential moving average)
    """
    if method == "historical":
        return returns.mean().values * TRADING_DAYS
    elif method == "ema":
        weights = np.exp(np.linspace(0, 1, len(returns)))
        weights /= weights.sum()
        ema_returns = (returns * weights[:, None]).sum()
        return ema_returns.values * TRADING_DAYS
    elif method == "capm":
        # Beta respecto al primer activo como proxy de mercado
        market = returns.iloc[:, 0]
        mu_m = market.mean() * TRADING_DAYS
        betas = returns.apply(lambda col: np.cov(col, market)[0, 1] / np.var(market))
        return (RISK_FREE_RATE_ANNUAL + betas * (mu_m - RISK_FREE_RATE_ANNUAL)).values
    return returns.mean().values * TRADING_DAYS


def compute_covariance(returns: pd.DataFrame, method: str = "sample") -> np.ndarray:
    """
    Calcula matriz de covarianza anualizada.
    method: 'sample' | 'ledoit_wolf' (shrinkage)
    """
    if method == "ledoit_wolf":
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf()
            lw.fit(returns)
            return lw.covariance_ * TRADING_DAYS
        except ImportError:
            pass
    return returns.cov().values * TRADING_DAYS


# ── Métricas de cartera ───────────────────────────────────────────────────────

def portfolio_performance(weights: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> dict:
    """Retorno, volatilidad y Sharpe de una cartera dados pesos."""
    ret = float(np.dot(weights, mu))
    vol = float(np.sqrt(weights @ sigma @ weights))
    sharpe = (ret - RISK_FREE_RATE_ANNUAL) / vol if vol > 0 else 0
    return {"return": ret, "volatility": vol, "sharpe": sharpe}


def compute_var(returns: pd.DataFrame, weights: np.ndarray,
                confidence: float = 0.95, method: str = "historical") -> dict:
    """Value at Risk (VaR) y Conditional VaR (CVaR)."""
    port_returns = returns.values @ weights
    if method == "historical":
        var = np.percentile(port_returns, (1 - confidence) * 100)
        cvar = port_returns[port_returns <= var].mean()
    else:  # paramétrico
        mu_p = port_returns.mean()
        sigma_p = port_returns.std()
        from scipy.stats import norm
        var = norm.ppf(1 - confidence, mu_p, sigma_p)
        cvar = mu_p - sigma_p * norm.pdf(norm.ppf(1 - confidence)) / (1 - confidence)
    return {
        "var_daily": float(var),
        "cvar_daily": float(cvar),
        "var_annual": float(var * np.sqrt(TRADING_DAYS)),
        "cvar_annual": float(cvar * np.sqrt(TRADING_DAYS)),
    }


def compute_drawdown(prices: pd.Series) -> dict:
    """Drawdown máximo histórico."""
    cum = (1 + prices.pct_change().dropna()).cumprod()
    roll_max = cum.cummax()
    drawdown = (cum - roll_max) / roll_max
    return {
        "max_drawdown": float(drawdown.min()),
        "drawdown_series": drawdown,
        "underwater": cum / roll_max - 1,
    }


def compute_beta(port_returns: pd.Series, market_returns: pd.Series) -> float:
    """Beta respecto a un benchmark."""
    cov = np.cov(port_returns, market_returns)
    return cov[0, 1] / cov[1, 1]


# ── Optimización ──────────────────────────────────────────────────────────────

def _neg_sharpe(weights, mu, sigma):
    p = portfolio_performance(weights, mu, sigma)
    return -p["sharpe"]


def _portfolio_vol(weights, mu, sigma):
    return portfolio_performance(weights, mu, sigma)["volatility"]


def _neg_return(weights, mu, sigma):
    return -portfolio_performance(weights, mu, sigma)["return"]


def optimize_portfolio(mu: np.ndarray, sigma: np.ndarray, n: int,
                        objective: str = "sharpe",
                        target_return: Optional[float] = None,
                        target_vol: Optional[float] = None,
                        allow_short: bool = False,
                        max_weight: float = 1.0) -> dict:
    """
    Optimiza cartera.
    objective: 'sharpe' | 'min_vol' | 'max_return' | 'target_return' | 'target_vol'
    """
    bounds = ((-1.0 if allow_short else 0.0, max_weight),) * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    x0 = np.ones(n) / n

    if objective == "sharpe":
        fun = _neg_sharpe
    elif objective == "min_vol":
        fun = _portfolio_vol
    elif objective == "max_return":
        fun = _neg_return
    elif objective == "target_return":
        constraints.append({
            "type": "eq",
            "fun": lambda w: portfolio_performance(w, mu, sigma)["return"] - target_return
        })
        fun = _portfolio_vol
    elif objective == "target_vol":
        constraints.append({
            "type": "ineq",
            "fun": lambda w: target_vol - portfolio_performance(w, mu, sigma)["volatility"]
        })
        fun = _neg_return
    else:
        fun = _neg_sharpe

    result = minimize(
        fun, x0,
        args=(mu, sigma),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9}
    )

    if result.success or result.status == 4:
        w = result.x
        w = np.clip(w, 0 if not allow_short else -1, max_weight)
        w /= w.sum()
        perf = portfolio_performance(w, mu, sigma)
        return {"weights": w, "success": True, **perf}
    return {"weights": x0, "success": False, "message": result.message,
            **portfolio_performance(x0, mu, sigma)}


# ── Frontera Eficiente ────────────────────────────────────────────────────────

def compute_efficient_frontier(mu: np.ndarray, sigma: np.ndarray, n: int,
                                n_points: int = 60,
                                allow_short: bool = False,
                                max_weight: float = 1.0) -> pd.DataFrame:
    """Genera puntos sobre la frontera eficiente."""
    min_ret = mu.min()
    max_ret = mu.max()
    target_returns = np.linspace(min_ret * 0.8, max_ret * 1.1, n_points)

    frontier = []
    for tr in target_returns:
        res = optimize_portfolio(mu, sigma, n, objective="target_return",
                                  target_return=tr, allow_short=allow_short,
                                  max_weight=max_weight)
        frontier.append({
            "target_return": tr,
            "volatility": res["volatility"],
            "return": res["return"],
            "sharpe": res["sharpe"],
        })

    return pd.DataFrame(frontier)


def compute_random_portfolios(mu: np.ndarray, sigma: np.ndarray, n: int,
                               n_portfolios: int = 3000) -> pd.DataFrame:
    """Simula carteras aleatorias para visualización (nube de Monte Carlo)."""
    results = []
    for _ in range(n_portfolios):
        w = np.random.dirichlet(np.ones(n))
        p = portfolio_performance(w, mu, sigma)
        results.append(p)
    return pd.DataFrame(results)


# ── Backtesting simple ────────────────────────────────────────────────────────

def backtest_portfolio(prices: pd.DataFrame, weights: np.ndarray,
                        rebalance_freq: str = "ME") -> pd.DataFrame:
    """
    Backtesting simple con rebalanceo periódico.
    rebalance_freq: 'ME' (mensual), 'QE' (trimestral), 'YE' (anual), None (buy&hold)
    """
    returns = prices.pct_change().dropna()

    if rebalance_freq is None:
        port_returns = (returns * weights).sum(axis=1)
    else:
        port_returns_list = []
        grouped = returns.groupby(pd.Grouper(freq=rebalance_freq))
        for period, period_ret in grouped:
            if period_ret.empty:
                continue
            port_returns_list.append((period_ret * weights).sum(axis=1))
        port_returns = pd.concat(port_returns_list).sort_index()

    cum_returns = (1 + port_returns).cumprod()
    total_return = cum_returns.iloc[-1] - 1
    ann_return = (1 + total_return) ** (TRADING_DAYS / len(port_returns)) - 1
    ann_vol = port_returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = (ann_return - RISK_FREE_RATE_ANNUAL) / ann_vol if ann_vol > 0 else 0

    roll_max = cum_returns.cummax()
    drawdown = (cum_returns - roll_max) / roll_max

    return pd.DataFrame({
        "portfolio": cum_returns,
        "drawdown": drawdown,
        "daily_return": port_returns,
    }), {
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": drawdown.min(),
    }


# ── Proyección de precio objetivo ────────────────────────────────────────────

def project_price_target(prices: pd.Series, horizon_months: int = 12,
                           n_simulations: int = 500) -> dict:
    """
    Proyecta precio futuro con simulación Monte Carlo (GBM).
    Retorna percentiles y escenarios.
    """
    returns = np.log(prices / prices.shift(1)).dropna()
    mu_daily = returns.mean()
    sigma_daily = returns.std()
    S0 = prices.iloc[-1]

    n_steps = horizon_months * 21  # ~21 días bursátiles/mes
    dt = 1

    sims = np.zeros((n_simulations, n_steps))
    for i in range(n_simulations):
        Z = np.random.standard_normal(n_steps)
        sims[i, 0] = S0
        for t in range(1, n_steps):
            sims[i, t] = sims[i, t - 1] * np.exp(
                (mu_daily - 0.5 * sigma_daily**2) * dt + sigma_daily * np.sqrt(dt) * Z[t]
            )

    # Fechas futuras (mensuales)
    last_date = prices.index[-1]
    future_dates = pd.date_range(last_date, periods=n_steps, freq="B")

    p10 = np.percentile(sims, 10, axis=0)
    p25 = np.percentile(sims, 25, axis=0)
    p50 = np.percentile(sims, 50, axis=0)
    p75 = np.percentile(sims, 75, axis=0)
    p90 = np.percentile(sims, 90, axis=0)

    return {
        "dates": future_dates,
        "p10": p10, "p25": p25, "p50": p50, "p75": p75, "p90": p90,
        "simulations": sims[:50],  # Solo 50 para no saturar gráficos
        "S0": S0,
        "horizon_months": horizon_months,
        "target_bear": float(p25[-1]),
        "target_base": float(p50[-1]),
        "target_bull": float(p75[-1]),
        "target_optimist": float(p90[-1]),
    }
