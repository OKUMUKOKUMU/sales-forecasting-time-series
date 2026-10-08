"""The six forecasting models (plus an ensemble), each with the same interface:

    forecast(train_df, future_df) -> np.ndarray of point forecasts for future_df

train_df / future_df have columns: week (datetime), units, promo, post_promo, holiday, price_step.
future_df's 'units' are never used, only the future-known regressors (promo, holiday, price_step).
"""
import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
for _n in ("cmdstanpy", "prophet", "prophet.plot"):
    logging.getLogger(_n).disabled = True
SEASON = 52


def add_regressors(df):
    df = df.copy()
    df["post_promo"] = np.r_[0, df.promo.values[:-1]]
    df["price_step"] = (df.week >= "2024-07-01").astype(int)   # known business event: price increase
    return df


def fourier(weeks, K=6, period=52.18):
    t = (weeks - pd.Timestamp("2021-01-04")).dt.days.values / 7
    return np.column_stack([f(2 * np.pi * k * t / period) for k in range(1, K + 1) for f in (np.sin, np.cos)])


# 1. Seasonal naive: "same week last year" - the benchmark every model must beat
def seasonal_naive(train, future):
    h = len(future)
    last = train.units.values[-SEASON:]
    return np.resize(last, h)


# 2. Holt-Winters exponential smoothing (ETS): level + damped trend + yearly seasonality, on log scale
def holt_winters(train, future):
    m = ExponentialSmoothing(np.log(train.units.values), trend="add", damped_trend=True,
                             seasonal="add", seasonal_periods=SEASON).fit(optimized=True)
    return np.exp(m.forecast(len(future)))


# 3. Theta method: deseasonalised series = long-run linear trend + simple exponential smoothing
def theta(train, future):
    s = pd.Series(np.log(train.units.values), index=pd.RangeIndex(len(train)))
    m = ThetaModel(s, period=SEASON, deseasonalize=True, method="additive").fit()
    return np.exp(m.forecast(len(future)).values)


# 4. SARIMAX: regression on Fourier seasonality + promo/holiday/price regressors, with ARIMA errors
def sarimax(train, future, order=(0, 1, 2)):
    """d = 1 differencing handles the trend; trend="c" adds a drift (average weekly growth)."""
    cols = ["promo", "post_promo", "holiday", "price_step"]
    Xtr = np.column_stack([fourier(train.week), train[cols].values])
    Xf = np.column_stack([fourier(future.week), future[cols].values])
    m = SARIMAX(np.log(train.units.values), exog=Xtr, order=order, trend="c").fit(disp=False)
    return np.exp(m.forecast(len(future), exog=Xf))


# 5. Prophet: piecewise trend with changepoints + yearly seasonality + regressors
def prophet(train, future):
    for name in ("cmdstanpy", "prophet", "prophet.plot"):
        logging.getLogger(name).disabled = True
    from prophet import Prophet
    cols = ["promo", "post_promo", "holiday", "price_step"]
    m = Prophet(yearly_seasonality=10, weekly_seasonality=False, daily_seasonality=False,
                seasonality_mode="multiplicative", changepoint_prior_scale=.05)
    for c in cols:
        m.add_regressor(c, mode="multiplicative")
    m.fit(train.rename(columns={"week": "ds", "units": "y"})[["ds", "y"] + cols])
    return m.predict(future.rename(columns={"week": "ds"})[["ds"] + cols]).yhat.values


# 6. Gradient boosting (machine learning): predicts year-on-year growth vs the same week last
#    year, from calendar and regressor features. Only uses information available at forecast time.
def _gbm_frame(df, n_train):
    d = df.copy()
    d["lag52"] = d.units.shift(SEASON)
    d["yoy_log"] = np.log(d.units / d.lag52)
    hist_yoy = d.yoy_log.where(d.index < n_train)             # never look at future actuals
    d["recent_yoy"] = hist_yoy.shift(1).rolling(13, min_periods=4).mean()
    d["recent_yoy"] = d.recent_yoy.where(d.index < n_train).ffill()   # frozen at forecast origin
    d["woy"] = d.week.dt.isocalendar().week.astype(int)
    for c in ["promo", "post_promo", "holiday", "price_step"]:
        d[f"{c}_lag52"] = d[c].shift(SEASON)
    return d


GBM_FEATURES = ["recent_yoy", "woy", "promo", "post_promo", "holiday", "price_step",
                "promo_lag52", "post_promo_lag52", "holiday_lag52", "price_step_lag52"]


def gradient_boosting(train, future):
    full = pd.concat([train, future.assign(units=np.nan)], ignore_index=True)
    n = len(train)
    d = _gbm_frame(full, n)
    # recursive "same week last year": if a lag52 falls inside the forecast window, use earlier forecast
    tr = d.iloc[:n].dropna(subset=GBM_FEATURES + ["yoy_log"])
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=.05, max_depth=3,
                                      min_samples_leaf=8, random_state=0).fit(tr[GBM_FEATURES], tr.yoy_log)
    preds = []
    units = full.units.values.copy()
    for i in range(n, len(full)):
        row = d.iloc[[i]][GBM_FEATURES]
        lag = units[i - SEASON]
        p = lag * np.exp(m.predict(row)[0])
        units[i] = p
        preds.append(p)
    return np.array(preds)


MODELS = {
    "Seasonal naive": seasonal_naive,
    "Holt-Winters (ETS)": holt_winters,
    "Theta": theta,
    "SARIMAX + Fourier": sarimax,
    "Prophet": prophet,
    "Gradient boosting": gradient_boosting,
}
ENSEMBLE_MEMBERS = ["Holt-Winters (ETS)", "SARIMAX + Fourier", "Prophet", "Gradient boosting"]


def metrics(actual, pred, train_units):
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    e = actual - pred
    scale = np.mean(np.abs(train_units[SEASON:] - train_units[:-SEASON]))   # in-sample seasonal-naive MAE
    return {"MAE": np.mean(np.abs(e)), "RMSE": np.sqrt(np.mean(e ** 2)),
            "MAPE %": 100 * np.mean(np.abs(e) / actual), "MASE": np.mean(np.abs(e)) / scale,
            "Bias %": 100 * (pred.sum() - actual.sum()) / actual.sum()}   # + = over-forecast
