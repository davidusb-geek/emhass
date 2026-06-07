# The Darts time-series forecaster

EMHASS ships an optional time-series forecaster built on the
[Darts](https://unit8co.github.io/darts/) library. It uses a global
gradient-boosted time-series model (LightGBM) instead of the recursive
`scikit-learn` approach of the [machine learning forecaster](mlforecaster.md).

The forecaster is **target-agnostic**, so the same model serves three EMHASS
forecasts, each selected independently:

- the **load** forecast (`load_forecast_method: darts`);
- the **load cost** forecast (`load_cost_forecast_method: darts`);
- the **PV production price** forecast (`production_price_forecast_method: darts`).

It exists alongside `mlforecaster`, not in place of it. Pick whichever scores
best for your home — see [when to use it](#when-to-use-it) below.

## How it differs from the `mlforecaster`

| | `mlforecaster` | `darts` |
|---|---|---|
| Model | recursive `scikit-learn` regressors | single-shot LightGBM (Darts) |
| Horizon | predicts step-by-step, feeding predictions back | emits the whole horizon in one pass |
| Weather | calendar features only | optional weather **future covariates** |
| Uncertainty | point forecast | optional quantiles (e.g. P10/P50/P90) |
| Dependency | bundled | optional `emhass[darts]` extra |

Because it is single-shot, the Darts model does not accumulate recursive error
across a long horizon, and because it accepts *future covariates* it can use
weather that is known for the forecast window (temperature, cloud cover, etc.)
rather than just calendar periodicity.

## Installation

Darts and LightGBM are heavyweight, so they are an **optional extra**. EMHASS
installs and runs with its default forecaster without them; install the extra
only if you want to use the `darts` method:

```bash
pip install emhass[darts]
```

If you select `load_forecast_method: darts` without the extra installed, EMHASS
raises a clear error telling you to install it.

## Using it

Set the load forecast method to `darts` (in the configuration page, or
`config.json`):

```json
"load_forecast_method": "darts"
```

Then train and predict with the same endpoints as the `mlforecaster`:

- `forecast-model-fit` — train a model from Home Assistant history.
- `forecast-model-predict` — produce a forecast from the trained model.
- `forecast-model-tune` — a light hyper-parameter search.

The fitted model is saved as `<model_type>_darts.pkl` in the data folder (the
`mlforecaster` uses `<model_type>_mlf.pkl`), so both can coexist.

The same runtime parameters apply as for the `mlforecaster`
(`historic_days_to_retrieve`, `model_type`, `var_model`, `num_lags`,
`split_date_delta`), plus the Darts-specific options below.

### Darts-specific options

- `darts_quantiles`: a list of quantile levels for a probabilistic forecast,
  e.g. `[0.1, 0.5, 0.9]`. When set, the model is fit with a quantile likelihood
  and the **P50** (median) is used as the point forecast; the other quantiles are
  available for callers that want uncertainty bands. Leave empty (the default)
  for a single deterministic forecast.

- `darts_covariate_columns`: a list of extra column names to use as weather
  future covariates. Calendar features (hour, day-of-week, etc.) are always
  added automatically; these are *additional* numeric covariates the model can
  use. Defaults to empty (calendar-only), which works on the target sensor alone.

- `darts_non_negative`: whether the target is physically non-negative and the
  forecast should be floored at zero. Keep the default `true` for a load forecast
  (watts); set it `false` when fitting a **price** model, whose target can
  legitimately go negative (negative spot prices occur on grids such as Amber).

```{note}
Weather covariates are most useful when your dominant load is weather-driven
(a heat pump, air conditioning). Calendar-only forecasting already captures
daily and weekly habits and needs no extra data.
```

## Using it for the load cost and production price

The same forecaster also serves the **load cost** and **PV production price**
forecasts. Because the model is target-agnostic, you train it exactly like the
load model — just point `var_model` at your price sensor and give the model a
distinct `model_type` suffix so the pickle does not collide with the load model:

1. **Fit** a model on your price history with `forecast-model-fit`, passing:
   - `var_model`: the Home Assistant sensor holding the price/cost series
     (e.g. an Amber import-price sensor);
   - `model_type`: `<base>_load_cost` for the load cost model, or
     `<base>_prod_price` for the production price model — the saved pickle is
     `<base>_load_cost_darts.pkl` / `<base>_prod_price_darts.pkl`;
   - `darts_non_negative: false` if your price can go negative.

2. **Select** the method in the configuration:

   ```json
   "load_cost_forecast_method": "darts",
   "production_price_forecast_method": "darts"
   ```

At optimization time EMHASS loads the matching pickle, fetches a fresh recent
window of the price sensor from Home Assistant to anchor the model's lags, and
produces the price series over the horizon. Calendar covariates are always
exact; any weather covariates are reused where the window supplies them and
forward-filled otherwise.

```{note}
Price is dominated by the time-of-day and weekly cycle, so a weather covariate
helps it far less than it helps a weather-driven load — useful but a smaller
effect. The single-shot design means the price forecast holds its shape over a
multi-day horizon without the recursive flattening of an autoregressive model.
```

## When to use it

There is no universally best load forecaster — it depends on your home's load
shape and how weather-driven it is. The recommended approach is to try both
`mlforecaster` and `darts` against your own data and keep whichever has the
lower error.

A reasonable rule of thumb:

- **Weather-dominated load** (heat pump, A/C) and you can supply weather
  covariates → `darts` with `darts_covariate_columns` often wins.
- **Long horizons** (multi-day day-ahead) where a recursive model flattens
  towards the mean → the single-shot `darts` model tends to hold its shape.
- **Light, calendar-driven load** or you want zero extra dependencies →
  `mlforecaster` or the `typical` method is simpler and adequate.

## Sanity checking

The Darts forecaster exposes a `sanity_check` helper that flags a grossly
diverging forecast (non-finite/negative values, an implausible peak, or a
horizon mean far from the recent daily **median** of recorded history). It is a
defensive gate for callers that want to fall back to another method if a
forecast looks broken; it never silently alters the forecast.
