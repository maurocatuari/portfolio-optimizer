# Portfolio Optimizer Pro 📊

Sistema de optimización cuantitativa de carteras de inversión con interfaz web en Streamlit.

## Características

- **Descarga de datos** — Yahoo Finance con caché inteligente (evita re-descargas)
- **Optimización de cartera** — Frontera eficiente de Markowitz, Máximo Sharpe, Mínima Varianza
- **Simulación Monte Carlo** — Proyección de precios y valor de cartera
- **Métricas de riesgo** — VaR, CVaR, Drawdown, Beta, Volatilidad rodante
- **Backtesting** — Con rebalanceo mensual/trimestral/anual vs buy&hold
- **Visualizaciones** — 13 gráficos interactivos con Plotly

## Instalación rápida

```bash
# Clonar o descomprimir el proyecto
cd portfolio_app

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar
streamlit run app.py
```

La app queda disponible en `http://localhost:8501`

## Docker

```bash
docker build -t portfolio-optimizer .
docker run -p 8501:8501 portfolio-optimizer
```

## Estructura del proyecto

```
portfolio_app/
├── app.py                  # App principal Streamlit
├── requirements.txt
├── Dockerfile
├── .env.example            # Plantilla de variables de entorno
│
├── data/
│   └── downloader.py       # Descarga, caché y gestión de activos
│
├── models/
│   └── optimizer.py        # Optimización, métricas de riesgo, Monte Carlo
│
├── utils/
│   └── charts.py           # 13 gráficos Plotly
│
├── assets/
│   └── watchlist.json      # Lista de activos guardada
│
└── cache/                  # Caché local de precios (Parquet)
```

## Uso

1. Abrí la app en el navegador
2. Ingresá los tickers en el panel izquierdo (ej: `SPY, QQQ, AGG, GLD`)
3. Seleccioná el período y objetivo de optimización
4. Hacé clic en **Calcular Cartera**

### Tickers soportados

- **ETFs**: SPY, QQQ, IWM, AGG, GLD, VTI, etc.
- **Acciones**: AAPL, MSFT, GOOGL, etc.
- **Cryptos**: BTC-USD, ETH-USD
- **CEDEARs/ADRs**: YPF, GGAL, BMA, BBAR (en Yahoo como .BA o sin sufijo)
- **Bonos**: TLT, BND, SHY

## Cambiar fuente de datos

### Opción A: Alpha Vantage

```python
# En data/downloader.py
from data.downloader import download_alpha_vantage
import os

api_key = os.getenv("AV_API_KEY")
df = download_alpha_vantage("AAPL", api_key)
```

Requiere registrarse gratis en https://www.alphavantage.co/

### Opción B: Agregar scraping propio

```python
# data/downloader.py — agregar función custom
import requests
from bs4 import BeautifulSoup

def download_bolsar(ticker: str) -> pd.DataFrame:
    """Scraping de Bolsar.com para activos argentinos."""
    url = f"https://bolsar.info/historico/{ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=10)
    # ... parsear tabla HTML
    # Respetar robots.txt y agregar delay entre requests
```

### Opción C: CSV propio

```python
df = pd.read_csv("mis_precios.csv", index_col=0, parse_dates=True)
# La estructura esperada: index=fechas, columnas=tickers
```

## Modelos de retorno esperado

| Método | Descripción | Cuándo usar |
|--------|-------------|-------------|
| `historical` | Media histórica simple | Cuando hay datos suficientes (>2 años) |
| `ema` | Media exponencial ponderada | Da más peso a datos recientes |
| `capm` | Capital Asset Pricing Model | Cuando querés retornos ajustados por beta |

## Variables de entorno

Copiá `.env.example` a `.env`:

```bash
cp .env.example .env
# Editar con tu editor favorito
```

| Variable | Descripción |
|----------|-------------|
| `AV_API_KEY` | API key de Alpha Vantage (gratis en alphavantage.co) |

## Extensiones futuras

- [ ] Black-Litterman (combinar opiniones del usuario con el mercado)
- [ ] Rebalanceo con costos de transacción
- [ ] Soporte para activos en ARS (integración con BNA para tipo de cambio)
- [ ] Alertas de precio por email
- [ ] Exportar cartera a CSV/PDF

## Advertencias

- Yahoo Finance puede tener datos faltantes o errores ocasionales
- Los retornos pasados no garantizan resultados futuros
- Las proyecciones Monte Carlo son modelos estadísticos, no predicciones
- Para activos argentinos, verificar el ticker correcto en Yahoo (ej: `YPFD.BA`)
