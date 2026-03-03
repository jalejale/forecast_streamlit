"""
forecasting.py — Core forecasting logic module
Implements: Moving Average, Exponential Smoothing (SES/Holt/Holt-Winters), SARIMA, Auto ARIMA
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pmdarima as pm


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data(file) -> pd.DataFrame:
    """Load uploaded CSV or Excel and return a clean DataFrame."""
    name = file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(file)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file)
    else:
        raise ValueError("Unsupported file format. Please upload CSV or Excel.")
    return df


def prepare_series(df: pd.DataFrame, date_col: str, value_col: str,
                   freq: str = "MS") -> pd.Series:
    """Convert DataFrame columns to a proper DatetimeIndex time series."""
    df = df[[date_col, value_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    df = df.set_index(date_col)
    series = df[value_col].asfreq(freq)
    return series


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(actual: pd.Series, fitted: pd.Series) -> dict:
    """Return MAE, RMSE, MAPE for model evaluation."""
    a = actual.dropna()
    f = fitted.reindex(a.index).dropna()
    a, f = a.align(f, join="inner")
    mae = mean_absolute_error(a, f)
    rmse = np.sqrt(mean_squared_error(a, f))
    mape = np.mean(np.abs((a - f) / a.replace(0, np.nan))) * 100
    return {"MAE": round(mae, 3), "RMSE": round(rmse, 3), "MAPE (%)": round(mape, 2)}


# ─────────────────────────────────────────────────────────────────────────────
# MOVING AVERAGE
# ─────────────────────────────────────────────────────────────────────────────

def moving_average(series: pd.Series, window: int = 3,
                   periods: int = 12) -> dict:
    """Simple trailing moving average forecast."""
    fitted = series.rolling(window=window).mean()
    last_ma = series[-window:].mean()
    last_date = series.index[-1]
    freq = series.index.freq or pd.tseries.frequencies.to_offset("MS")
    future_idx = pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]
    forecast = pd.Series(last_ma, index=future_idx)

    # Very basic CI (±1 std of last window)
    std = series[-window:].std()
    lower = forecast - 1.96 * std
    upper = forecast + 1.96 * std

    metrics = compute_metrics(series, fitted)
    return {
        "fitted": fitted,
        "forecast": forecast,
        "lower": lower,
        "upper": upper,
        "metrics": metrics,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SIMPLE EXPONENTIAL SMOOTHING
# ─────────────────────────────────────────────────────────────────────────────

def ses_forecast(series: pd.Series, alpha: float = 0.3,
                 periods: int = 12) -> dict:
    model = SimpleExpSmoothing(series, initialization_method="estimated")
    fit = model.fit(smoothing_level=alpha, optimized=False)
    fc = fit.forecast(periods)
    std_resid = np.std(fit.resid)
    lower = fc - 1.96 * std_resid
    upper = fc + 1.96 * std_resid
    return {
        "fitted": fit.fittedvalues,
        "forecast": fc,
        "lower": lower,
        "upper": upper,
        "metrics": compute_metrics(series, fit.fittedvalues),
    }


# ─────────────────────────────────────────────────────────────────────────────
# HOLT'S LINEAR (DOUBLE EXPONENTIAL SMOOTHING)
# ─────────────────────────────────────────────────────────────────────────────

def holt_forecast(series: pd.Series, periods: int = 12) -> dict:
    model = ExponentialSmoothing(series, trend="add",
                                  initialization_method="estimated")
    fit = model.fit(optimized=True)
    fc = fit.forecast(periods)
    std_resid = np.std(fit.resid)
    lower = fc - 1.96 * std_resid
    upper = fc + 1.96 * std_resid
    return {
        "fitted": fit.fittedvalues,
        "forecast": fc,
        "lower": lower,
        "upper": upper,
        "metrics": compute_metrics(series, fit.fittedvalues),
    }


# ─────────────────────────────────────────────────────────────────────────────
# HOLT-WINTERS (TRIPLE EXPONENTIAL SMOOTHING)
# ─────────────────────────────────────────────────────────────────────────────

def holtwinters_forecast(series: pd.Series, seasonal_periods: int = 12,
                          trend: str = "add", seasonal: str = "add",
                          periods: int = 12) -> dict:
    model = ExponentialSmoothing(
        series,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
        initialization_method="estimated",
    )
    fit = model.fit(optimized=True)
    fc = fit.forecast(periods)
    std_resid = np.std(fit.resid)
    lower = fc - 1.96 * std_resid
    upper = fc + 1.96 * std_resid
    return {
        "fitted": fit.fittedvalues,
        "forecast": fc,
        "lower": lower,
        "upper": upper,
        "metrics": compute_metrics(series, fit.fittedvalues),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SARIMA
# ─────────────────────────────────────────────────────────────────────────────

def sarima_forecast(series: pd.Series,
                    order: tuple = (1, 1, 1),
                    seasonal_order: tuple = (1, 1, 1, 12),
                    periods: int = 12) -> dict:
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False)
    fc_obj = fit.get_forecast(steps=periods)
    fc = fc_obj.predicted_mean
    ci = fc_obj.conf_int(alpha=0.05)
    return {
        "fitted": fit.fittedvalues,
        "forecast": fc,
        "lower": ci.iloc[:, 0],
        "upper": ci.iloc[:, 1],
        "metrics": compute_metrics(series, fit.fittedvalues),
        "summary": fit.summary().as_text(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# AUTO ARIMA  (pmdarima)
# ─────────────────────────────────────────────────────────────────────────────

def auto_arima_forecast(series: pd.Series,
                        seasonal: bool = True,
                        m: int = 12,
                        periods: int = 12,
                        stepwise: bool = True,
                        information_criterion: str = "aic") -> dict:
    """
    Automatically select the best ARIMA/SARIMA orders using pmdarima.
    Equivalent to R's auto.arima() from the `forecast` package.
    """
    model = pm.auto_arima(
        series,
        seasonal=seasonal,
        m=m,
        stepwise=stepwise,
        information_criterion=information_criterion,
        suppress_warnings=True,
        error_action="ignore",
        trace=False,
    )

    # In-sample fitted values
    fitted_vals = series - model.resid()
    fitted = pd.Series(fitted_vals, index=series.index)

    # Forecast with confidence interval
    fc_array, conf_int = model.predict(n_periods=periods, return_conf_int=True, alpha=0.05)

    last_date = series.index[-1]
    freq = series.index.freq or pd.tseries.frequencies.to_offset("MS")
    future_idx = pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]

    fc   = pd.Series(fc_array,       index=future_idx)
    lower = pd.Series(conf_int[:, 0], index=future_idx)
    upper = pd.Series(conf_int[:, 1], index=future_idx)

    # Selected order information
    order         = model.order
    seasonal_order = model.seasonal_order
    order_str = (
        f"Best order: ARIMA({order[0]},{order[1]},{order[2]})"
        f"×({seasonal_order[0]},{seasonal_order[1]},{seasonal_order[2]},{seasonal_order[3]})"
        if seasonal else
        f"Best order: ARIMA({order[0]},{order[1]},{order[2]})"
    )

    return {
        "fitted":    fitted,
        "forecast":  fc,
        "lower":     lower,
        "upper":     upper,
        "metrics":   compute_metrics(series, fitted),
        "summary":   model.summary().as_text(),
        "order_str": order_str,
        "order":     order,
        "seasonal_order": seasonal_order,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DECOMPOSITION
# ─────────────────────────────────────────────────────────────────────────────

def decompose_series(series: pd.Series, model: str = "additive",
                     period: int = 12):
    result = seasonal_decompose(series, model=model, period=period,
                                extrapolate_trend="freq")
    return result
