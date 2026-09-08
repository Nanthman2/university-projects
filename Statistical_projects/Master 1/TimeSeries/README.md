# Time Series — Nottingham Monthly Temperatures (1920–1939)

**Course:** Statistics coursework, UCLouvain (Master 1)

## 🎯 Objective

Model and **forecast the monthly mean temperature in Nottingham** (1920–1939), capturing the series' seasonality, and benchmark a seasonal ARIMA (SARIMA) model against Holt-Winters exponential smoothing.

## 🧩 Approach

- Seasonal differencing to remove trend/seasonality and reach stationarity
- Model identification via ACF/PACF analysis
- Model selection across SARIMA candidates via AIC comparison
- Residual diagnostics, including the **Ljung-Box test** for remaining autocorrelation
- Forecasting with prediction intervals, benchmarked against **Holt-Winters** exponential smoothing

## 📂 Repository contents

| File | Description |
|---|---|
| `Time_series.pdf` | Full report: methodology, model selection, diagnostics and forecasts |

> This is a **report-only** project — no separate code notebook is included here; the full analysis (including model output) is presented in the PDF report.
