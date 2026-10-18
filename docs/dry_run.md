# Dry run: plan without replacing the live plan

A dry run solves an optimization exactly as a live run does and returns the plan in the response. It keeps nothing: the live plan, the published entities, and the next live solve are as if the dry run never happened. Use it for what-ifs ("what does this import cap cost?", "does the car still reach 80 % by 17:00?") and to ask EMHASS for a plan from an outside coordinator, on the same instance that runs your live MPC.

Available from EMHASS 0.18.7.

## How to call it

Send the body you would send to `/action/<action>` to `/api/v1/dry-run/<action>`. The actions are `perfect-optim`, `dayahead-optim` and `naive-mpc-optim`.

For example, the live MPC call with a per-step import cap that leaves room for an EV charge:

```bash
curl -s -H 'Content-Type: application/json' -X POST \
  -d '{"prediction_horizon": 24, "soc_init": 0.5, "soc_final": 0.6,
       "maximum_power_from_grid": [9000, 9000, 9000, 9000, 2000, 2000, 2000, 2000, 9000, 9000, 9000, 9000,
                                   9000, 9000, 9000, 9000, 9000, 9000, 9000, 9000, 9000, 9000, 9000, 9000]}' \
  http://localhost:5000/api/v1/dry-run/naive-mpc-optim
```

An EMHASS older than 0.18.7 answers 404 at this URL. It never runs a live solve in place of a dry run.

## The response

| HTTP | `status` | Meaning |
| --- | --- | --- |
| 200 | `ok`, `infeasible` or `error` | The solver ran. `status` is what `/api/v1/last-run` would report; only an `ok` plan would have replaced `/api/v1/plan`. |
| 400 | `not-run` | The request was refused, or the run stopped before the solver (for example on a rejected forecast). `errors` says why. |
| 500 | `not-run` | The run failed. `errors` says why. |

The JSON body ([schema](https://github.com/davidusb-geek/emhass/blob/master/docs/api/v1/dry-run.schema.json)) has:

- `dry_run` (always `true`) and `action`;
- `status`;
- when the solver ran, the run record that `/api/v1/last-run` would have served: `timestamp`, `stage_times`, `duration_total_seconds`, `emhass_version`, `schema_version`, `infeasible`, `error_message`;
- `plan`: the records that `/api/v1/plan` would have served, with each row's `optim_status`. It is returned whatever the status, so you can see what an infeasible what-if gave. It is `null` when nothing was solved;
- `warnings` and `errors`: the ones this request logged, and no other request's.

## What a dry run leaves as it was

A dry run reads every cache, model and file a live run reads, and writes none of them:

- no results CSV (`opt_res_latest.csv` or a dated file);
- no entity files, and the continual-publish loop is not started or fed;
- no change to `/api/v1/plan` or `/api/v1/last-run`, in memory or on disk;
- no publish to Home Assistant;
- the in-memory `OptimizationCache` is not read or written. The next live solve gets the same cached problem, configuration and warm-start values. The dry run itself solves from a cold start;
- the weather caches (the Open-Meteo JSON caches and `weather_forecast_data.pkl`) are not saved, refreshed or deleted. A stale cache is fetched afresh for the dry run only;
- a stale adjusted-PV model is refitted in memory, as the live run would refit it, and not saved. Each dry run refits it again until the next live run saves a new one;
- battery identification does not run;
- the page's results table (`injection_dict.pkl`) is not saved;
- its log records do not go to the action log. They go to the console and to the response.

## What it does not cover

- Forecast services are called as in a live run. Open-Meteo, Solcast and solar.forecast fetches count against their quotas, and Solcast's daily call counter counts them. For repeatable what-ifs, pass the forecasts as lists, as in [Passing data](passing_data.md).
- Home Assistant is read, as in a live run.
- The machine learning actions (`forecast-model-fit` and the others), `publish-data`, `weather-forecast-cache` and the other actions have no dry run.
- The HTTP status of a live `/action/<action>` request is still read from the shared action log. A dry run cannot change it, because its records stay out of that log.

## What is refused

So that no caller gets a live run it did not ask for, and no caller believes something was saved:

- `dry_run` as a runtime parameter of `/action/<action>`, or of the command line, unless it is `false`: 400, and nothing runs. Older releases ignore the parameter and run live, so use the dry-run URL.
- On the dry-run URL: an action without a dry run; `dry_run` set to anything but `true`; `entity_save` or `weather_forecast_cache` set to true (they ask for a write); a body that is not a JSON object. Each gives 400, and nothing runs. `weather_forecast_cache_only` is allowed: it only reads.
