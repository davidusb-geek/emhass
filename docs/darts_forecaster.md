# The Darts time-series load forecaster

EMHASS ships an optional second load forecaster built on the
[Darts](https://unit8co.github.io/darts/) library. It uses a global
gradient-boosted time-series model (LightGBM) instead of the recursive
`scikit-learn` approach of the [machine learning forecaster](mlforecaster.md).

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
`split_date_delta`), plus two Darts-specific options.

### Darts-specific options

- `darts_quantiles`: a list of quantile levels for a probabilistic forecast,
  e.g. `[0.1, 0.5, 0.9]`. When set, the model is fit with a quantile likelihood
  and the **P50** (median) is used as the load forecast; the other quantiles are
  available for callers that want uncertainty bands. Leave empty (the default)
  for a single deterministic forecast.

- `darts_covariate_columns`: a list of extra column names to use as weather
  future covariates. Calendar features (hour, day-of-week, etc.) are always
  added automatically; these are *additional* numeric covariates the model can
  use. Defaults to empty (calendar-only), which works on the load sensor alone.

```{note}
Weather covariates are most useful when your dominant load is weather-driven
(a heat pump, air conditioning). Calendar-only forecasting already captures
daily and weekly habits and needs no extra data.
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
