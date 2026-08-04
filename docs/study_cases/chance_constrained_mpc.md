# From load forecasting to chance-constrained MPC

> **Type:** Explanation - understanding-oriented. A retrospective study linking held-out forecast quality to battery-control outcomes.

```{important}
This page describes a standalone research prototype, not a released EMHASS
feature. As of EMHASS 0.18.0, `naive-mpc-optim` optimizes one forecast path.
The rolling, `q10`, `q20`, and hard finite-scenario controllers below are
experiment labels, not runtime parameters or configuration options.
```

This study connects two questions that are often evaluated separately:

1. Which load forecasters produced the strongest held-out predictions?
2. Did those improvements change the resulting energy-management decisions?

## Data and evaluation protocol

The target was the household demand that EMHASS does not schedule directly,
stored as `sensor.power_load_no_var_loads` in `data/long_train_data.pkl`. The
series contains half-hourly data from 2024-02-28 through 2025-02-27. The final 14
local days, 2025-02-14 through 2025-02-27 in Europe/Paris, supplied 672 measured
test values.

The historical target contained 336 missing values, all before the final test
period. They were linearly interpolated before fitting; the 672 test targets
contained no missing values.

Two forecast views used those same 672 targets:

- **Day-ahead:** 14 non-overlapping forecasts made at local midnight, each with
  48 half-hour leads.
- **Rolling one-step:** a new remaining-day forecast every 30 minutes, scored at
  the first lead before the next measurement became available.

Model fitting, epoch selection, and uncertainty calibration excluded the final
test interval within each recorded run. The same holdout had nevertheless
informed the wider sequence of model-development experiments, so this is a
retrospective comparison rather than an untouched confirmatory benchmark.

## Forecast leaders and control-study forecasters

This page includes two additional leading zero-shot models from the full forecast
benchmark and the four forecasters with stored rolling-controller results. Granite
PatchTST-FM and Toto 2.0 are forecast-only here because the optimization replay
predated their evaluation and was not rerun, not because of licensing: both
recorded checkpoints use Apache-2.0 licenses.

FlowState was the strongest point forecaster with a stored control replay. EMA
was the strongest fitted ensemble with a replay, Gradient Boosting provided the
classical model reference, and the Gaussian Transformer supplied a deliberately
probabilistic extension for the quantile experiment.

### Granite PatchTST-FM (forecast only)

Granite PatchTST-FM used IBM's pretrained 257,895,552-parameter PatchTST-style
encoder. Its 8,192-position window left up to 8,064 observed inputs for this
48-step forecast request, and it emitted point and quantile forecasts directly.
No household-specific weights were fitted or updated.

Context length and official point-versus-median output were selected on the 14
days immediately before the final test. The fixed search compared 512, 1,024,
2,048, 4,096, and 8,064 historical values and selected the 2,048-step official
point output. That configuration then produced the stored 48-step day-ahead test
forecast. No rolling-origin forecast or optimizer replay was recorded.

Before that grid was frozen, exploratory 8,064-step point and median diagnostics
had already been viewed on the final holdout. They did not enter the numerical
selector, but they mean the canonical Granite result is reproducible iterative
evidence rather than an untouched confirmatory test.

### Toto 2.0 313M (forecast only)

Toto used Datadog's pretrained 312,684,608-parameter decoder-only patched
Transformer. It received 4,096 historical values and decoded the complete
48-step q50 path directly, without household-specific fitting or adaptation.

The benchmark evaluated the public 22M, 313M, and 1B checkpoint sizes; 313M had
the lowest observed test RMSE. Because that size comparison used the shared test
period rather than a separate validation selector, its leading status is
retrospective. No rolling-origin forecast or optimizer replay was recorded.

### FlowState r1.1

FlowState is a pretrained 18,524,928-parameter state-space forecaster with a
continuous functional-basis decoder. It received the checkpoint's required
4,096 half-hour values, approximately 85 days of history, and emitted the full
48-step horizon directly.

No household-specific fitting or weight update was performed. The checkpoint
produced nine native quantile paths from q10 through q90. Its q50 path was the
point forecast; q10 and q20 supplied the lower no-export bounds.

### EMA multiscale ensemble

Each member of the EMA ensemble was a 20,865-parameter multiscale
GridPatchTransformer. It transformed seven days, or 336 values, into:

- 48 seasonal tokens containing the seven historical values aligned to each
  future half-hour slot;
- 24 length-4 tokens from the latest 96 measurements; and
- 12 length-8 tokens from the same recent window.

The 84 tokens passed through two width-32 Transformer encoder layers with four
attention heads. A residual head predicted a direct 48-step correction to the
previous-day profile using known-future calendar encodings.

Training minimized raw-watt-equivalent mean squared error with AdamW. A 28-day
pre-test validation interval and seeds 17, 42, and 89 selected exponential
moving-average weights with decay 0.9 after epoch 2. Three fresh final members
were then fitted on all available pre-test windows. Their elementwise arithmetic
mean formed the point forecast used by the optimizer.

For uncertainty, the three individual seed predictions were reused as paths.
This seed spread was never calibrated as a predictive distribution. With only
three values at each lead, the empirical q10 and q20 lower order statistics both
selected the minimum prediction.

### Gradient Boosting

The classical model used `skforecast.ForecasterRecursive` around
`GradientBoostingRegressor`. It received 336 contiguous load lags plus six
calendar features, used learning rate 0.1 and random seed 42, and predicted one
step at a time. Each prediction was fed back recursively until all 48 day-ahead
leads were available.

Uncertainty calibration used a separate model fitted only on data before
2025-01-16 23:00 UTC. Its next 28 complete daily forecasts produced out-of-sample
pre-test residual trajectories. The final point model was then fitted once on all history
available before the final test boundary. Each calibration residual trajectory
was added to a rolling point forecast and negative loads were clipped to zero. At
each lead, q10 selected the third-lowest and q20 the sixth-lowest of the 28
residual-adjusted values.

### Gaussian multiscale Transformer

The Gaussian model reused the EMA ensemble's seven-day multiscale token
architecture, not its fitted weights, and replaced the point head with per-lead
mean and bounded log-variance heads. Each member had 20,898 parameters and was
trained with Gaussian negative log likelihood in a globally standardized target
coordinate.

A fresh seed-42 search selected epoch 2 EMA-0.9 weights by the lowest Gaussian
NLL on the 28-day pre-test validation interval. Three fresh final members were
then trained on all pre-test windows with seeds 17, 42, and 89. Their means and
variances were combined with the law of total variance, retaining both predicted
variance and between-member mean spread.

Nine marginal Gaussian q10-q90 paths were derived from the combined mean and
variance. One quantile rank was held across each remaining-day path. This is a
reproducible perfect-rank-dependence assumption, not a learned temporal error
covariance. Negative quantile values were clipped to zero.

## Held-out forecast results

RMSE measures forecast error in watts and emphasizes large misses:

$$
\operatorname{RMSE} = \sqrt{\operatorname{mean}((L-\hat{L})^2)}.
$$

**Day-ahead RMSE** applies this definition to all 48 leads from each of the 14
local-midnight forecasts. The 14 complete profiles produce 672 residuals, so
short- and long-horizon errors both contribute to the score.

**Rolling one-step RMSE** scores only the first lead from every forecast updated
at 30-minute intervals. Let $\hat{L}_{t,1}$ be the first value forecast for test
interval $t$ using measurements available before that interval. Then

$$
\operatorname{RMSE}_{\mathrm{rolling},1}
= \sqrt{\frac{1}{672}\sum_{t=1}^{672}(L_t-\hat{L}_{t,1})^2}.
$$

For example, the 10:00 forecast uses measurements available before 10:00 and is
scored against measured load at 10:00. At 10:30, the model includes the newly
observed 10:00 load, forecasts again, and only its 10:30 value enters this metric.
The optimizer still uses the full remaining-day forecast at each solve; rolling
one-step RMSE measures immediate forecast accuracy, not the accuracy of that
complete path.

NMSE divides squared error by measured-load signal power:

$$
\operatorname{NMSE}
= \frac{\operatorname{mean}((L-\hat{L})^2)}{\operatorname{mean}(L^2)}.
$$

The archived denominator for these 672 test timestamps is
`mean(actual^2) = 3,291,888.99 W^2`. Every non-archived NMSE below is a direct
arithmetic conversion of stored full-precision RMSE using that fixed denominator;
no forecasts were rerun.

| Forecaster | Point path | Day-ahead RMSE (W) | Day-ahead NMSE | Rolling one-step RMSE (W) | Rolling one-step NMSE | Stored optimizer replay |
|---|---|---:|---:|---:|---:|---|
| Granite PatchTST-FM | Official point output | **690.77** | **0.145** | Not evaluated | Not evaluated | No |
| FlowState r1.1 | Native q50 | 700.33 | 0.149 | **564.58** | **0.097** | Yes |
| Toto 2.0 313M | Native q50 | 717.21 | 0.156 | Not evaluated | Not evaluated | No |
| EMA multiscale ensemble | Mean of three EMA members | 729.17 | 0.162 | 715.12 | 0.155 | Yes |
| Gradient Boosting | Recursive point forecast | 735.36 | 0.164 | 591.61 | 0.106 | Yes |
| Gaussian multiscale Transformer | Combined Gaussian mean | 764.13 | 0.177 | 757.58 | 0.174 | Yes |

The validation-selected canonical Granite path had the lowest day-ahead RMSE and
NMSE in this table, followed by FlowState and Toto. A validation-rejected Granite
PCA-blend diagnostic later reached 681.31 W and 0.141, but it is not promoted here
because doing so after observing the test would introduce post-selection bias.
FlowState was the best model with a complete control replay and had the lowest
rolling one-step error. Among the remaining replayed models, Gradient Boosting
ranked second at the rolling first lead, showing why a midnight-only ranking does
not fully describe a controller that refreshes every half-hour. EMA and the
Gaussian model changed relatively little between the two views, with the
Gaussian mean changing least; the Gaussian model traded point accuracy for an
explicit variance estimate. Granite and Toto cannot be ranked on rolling
one-step accuracy because those forecasts were not recorded.

The archived forecast leaderboard lists Gradient Boosting at 733.59 W RMSE and
0.163 NMSE. Its timestamped optimizer input was regenerated in a different
software environment and reached 735.36 W and 0.164 instead. The latter is used
throughout this page because it is the path actually passed to the stored
optimization experiment.

## Optimizers compared

The **rolling MPC** is the recorded **30-minute rolling point-path MPC**: it
rebuilds a forecast from the most recent measurements, solves the remaining-day
problem, executes one action, and repeats. It is not a moving average of
measurements. The EMA and Gaussian point paths are arithmetic ensemble means,
while all four forecasters refresh their inputs from the latest available
measurements.

The stored uncertainty runs used q10 and q20; no other quantile level is reported
or inferred.

| Controller family | Update pattern | Forecast used for cost | Treatment of load uncertainty |
|---|---|---|---|
| Once-daily deterministic EMHASS baseline | One 48-step plan at local midnight | One midnight point path | None |
| 30-minute rolling point-path MPC | Reforecast and solve every half-hour; execute the first action | Current model-specific point path | No-export enforced against the point path |
| Lower-quantile chance MPC | Same rolling measurement feedback and first-action execution | Same current point path | Net discharge additionally bounded by q10 or q20 at every lead |

The point path was FlowState q50, the arithmetic mean of the three EMA members,
the Gradient Boosting point forecast, or the Gaussian ensemble mean. At each
rolling solve the horizon covered the rest of the local day, shrinking from 48
steps at midnight to one step at 23:30.

### Chance-constraint meaning

Let $u_t$ be net battery discharge and $L_t$ uncertain household load in the
zero-PV battery experiment. No-export requires

$$
L_t-u_t \geq 0.
$$

Requiring this at timestep $t$ with probability at least $1-\alpha$ gives

$$
\Pr(L_t \geq u_t) \geq 1-\alpha,
$$

with the lower-quantile deterministic bound

$$
u_t \leq q_{\alpha}(L_t).
$$

The q10 controller is therefore nominally a 90% per-timestep no-export
constraint and q20 is nominally 80%, provided the quantiles are calibrated. This
does not imply 90% or 80% probability that the complete horizon is export-free.

## Optimization setup and scoring

The once-daily baseline called EMHASS's `Optimization.perform_optimization`. The
rolling point and quantile controllers used a separate CVXPY prototype. The
comparable controller result is the battery-only, zero-PV case:

| Item | Value |
|---|---|
| Battery | 5 kWh nominal |
| Power | 1 kW nominal charge/discharge settings; 0.95 kW effective AC discharge bound in the rolling prototype |
| Efficiency | 0.95 for charge and discharge |
| SOC | Initial and target 0.6; hard bounds 0.3-0.9 |
| PV and deferrable load | Disabled |
| Import tariff | 0.1419 EUR/kWh off-peak; 0.1907 EUR/kWh peak |
| Grid limit | 9 kW |
| Battery degradation cost | Zero |

In the rolling prototype, SOC bounds, power limits, and charge/discharge
exclusivity were hard. No-export was hard against the supplied point path, with
q10 or q20 adding a stricter lower-load bound in the chance controllers. Grid
overflow and terminal SOC used penalized slack variables and were soft. Every
stored rolling run nevertheless ended at SOC 0.6 with zero overflow.

Execution used causal safety recourse. Requested discharge was clipped to
measured load and available SOC, while charging was clipped at maximum SOC,
before updating the battery state. **Safety adjustment** is the absolute energy
difference between the requested and realized action; an **intervention** is a
half-hour step where that difference exceeded the numerical tolerance. This
layer did not cap grid import. The stored replays observed zero realized
grid-limit and export violations.

The once-daily replays ended with different SOC values. Their comparable cost
therefore credits or debits terminal energy at off-peak replacement cost,
including charge loss:

```text
terminal_credit = (terminal_SOC - 0.6) * 5 kWh * 0.1419 EUR/kWh / 0.95
adjusted_cost = cash_bill - terminal_credit
```

The once-daily-to-rolling comparison changes forecast origin and history,
horizon, feedback cadence, and optimizer implementation. Its difference must
not be attributed to measurement feedback alone. Comparing q10 or q20 with the
rolling point controller is cleaner because those runs share the same rolling
protocol and differ only in the lower no-export bound.

## Economic optimizer results

The table reports terminal-SOC-adjusted cost over all 14 test days. All rolling
controllers ended at SOC 0.6, so their adjusted cost equals their cash bill. The
perfect-load oracle cost 79.7017 EUR and the no-battery reference cost 81.8763
EUR.

| Forecaster | Once-daily EMHASS baseline (EUR) | 30-minute rolling point MPC (EUR) | Chance q10 MPC (EUR) | Chance q20 MPC (EUR) |
|---|---:|---:|---:|---:|
| Granite PatchTST-FM | Not evaluated | Not evaluated | Not evaluated | Not evaluated |
| FlowState r1.1 | 79.9507 | **79.7122** | 79.7427 | 79.7306 |
| Toto 2.0 313M | Not evaluated | Not evaluated | Not evaluated | Not evaluated |
| EMA multiscale ensemble | 79.9642 | **79.7018** | 79.7028 | 79.7028 |
| Gradient Boosting | 80.0804 | **79.7091** | 81.1723 | 80.6132 |
| Gaussian multiscale Transformer | 80.0007 | **79.7017** | 80.4819 | 79.8438 |

Every evaluated rolling point result was lower than its once-daily comparator, by
0.2385-0.3713 EUR, but the protocol differences above prevent a pure feedback
claim. More importantly, the four rolling point controllers all came within
0.0105 EUR of the perfect-load oracle despite materially different forecast
errors.

No q10 or q20 controller beat its corresponding rolling point controller. q20
was cheaper than q10 whenever the two bounds differed because it permitted more
battery discharge. EMA did not change: three seed paths made q10 and q20 the
same minimum order statistic.

## Operational optimizer results

Cost alone hides how strongly each controller relied on the causal safety layer.
The rolling point controllers produced:

| Forecaster point path | Realized throughput (kWh) | Safety adjustment (kWh) | Interventions out of 672 |
|---|---:|---:|---:|
| FlowState q50 | 136.30 | **3.589** | **60** |
| EMA arithmetic mean | 136.95 | 11.914 | 80 |
| Gradient Boosting point | 136.50 | 5.919 | 73 |
| Gaussian ensemble mean | **136.96** | 12.259 | 80 |

Detailed once-daily replay retained 8.31 kWh of adjustment for FlowState, 8.43
kWh for EMA, and 14.00 kWh for Gradient Boosting. The supplemental Gaussian
once-daily baseline retained its adjusted cost but not detailed throughput or
recourse fields.

For chance MPC, **actual bound coverage** is the fraction of measured first-step
loads at or above the selected lower bound. It uses all 672 executed steps and
is not joint-horizon coverage.

| Forecaster | Controller | Actual bound coverage | Realized throughput (kWh) | Safety adjustment (kWh) | Interventions |
|---|---|---:|---:|---:|---:|
| FlowState | q10 | 83.33% | 134.38 | 1.235 | 24 |
| FlowState | q20 | 76.19% | 135.14 | 1.771 | 31 |
| EMA ensemble | q10 | 49.26% | 136.89 | 10.131 | 78 |
| EMA ensemble | q20 | 49.26% | 136.89 | 10.131 | 78 |
| Gradient Boosting | q10 | 99.11% | 62.83 | **0.000** | **0** |
| Gradient Boosting | q20 | 95.54% | 89.71 | 0.031 | 1 |
| Gaussian Transformer | q10 | 97.47% | 89.17 | 0.274 | 6 |
| Gaussian Transformer | q20 | 87.65% | 128.01 | 1.451 | 24 |

The nominal q10 and q20 coverage targets were 90% and 80%. FlowState missed both
targets, EMA seed spread missed them severely, and the Gradient Boosting and
Gaussian bounds over-covered. The over-covering Gradient Boosting and Gaussian
q10 bounds nearly eliminated safety intervention, but they also suppressed
battery use and raised cost. A coverage miss did not always trigger intervention
because the optimized discharge could remain below the selected bound.

## Additional stored sensitivities

The experiment also retained a fourth controller, **hard finite-scenario rolling
MPC**. It minimized equal-weight expected cost with one shared battery-action
sequence and required no-export against every supplied path. It was not a
scenario tree, had no scenario-dependent planned future recourse or CVaR, and is
not part of the requested three controller families. Each executed first action
still passed through the same causal safety replay.

| Forecaster | Supplied paths | Adjusted cost (EUR) | Realized throughput (kWh) |
|---|---:|---:|---:|
| FlowState | 9 | 79.7413 | 134.46 |
| EMA ensemble | 3 | 79.7028 | 136.89 |
| Gradient Boosting | 28 | 81.4817 | 47.42 |
| Gaussian Transformer | 9 | 80.4829 | 89.30 |

Broad or poorly calibrated scenario sets made hard protection very conservative.
This was most visible for Gradient Boosting and the Gaussian model. Their q20
controllers recovered throughput to 89.71 and 128.01 kWh, respectively, while
remaining more expensive than rolling point MPC.

A separate default-like case used measured PV, no battery, and a 3 kW deferrable
load requiring four hours of operation per day. Once-daily, rolling point, and
hard finite-scenario control all delivered 168 kWh and produced the same 75.4399
EUR bill. Chance controllers were not run because that case had no uncertain
battery no-export constraint. This negative result shows that forecast
uncertainty has no economic value when it does not change the active scheduling
decision.

## Overall conclusions

1. **Canonical Granite led day-ahead forecasting, while FlowState led the
   end-to-end set.** Granite reached 690.77 W RMSE and 0.145 NMSE, followed by
   FlowState at 700.33 W and 0.149 and Toto at 717.21 W and 0.156. Of those three
   forecast leaders, only FlowState had stored rolling and optimizer results; it
   also had the lowest rolling one-step RMSE/NMSE and the smallest rolling point
   safety adjustment.
2. **Stored rolling configurations had much lower oracle regret but barely
   changed the total bill.** Their adjusted costs were only 0.2385-0.3713 EUR
   below the once-daily comparators over 14 days, or approximately 0.30-0.46%.
   The perfect-load oracle had only 2.1746 EUR of total saving available relative
   to the no-battery reference. Once-daily control already captured approximately
   83-89% of that opportunity; the stored rolling configurations captured more
   than 99.5%.
3. **Forecast rank did not translate directly into bill rank.** The Gaussian
   rolling point controller matched the oracle to stored precision and EMA was
   only 0.00014 EUR higher, while both required about three times FlowState's
   safety adjustment. Economic cost alone therefore understated operational
   fragility.
4. **Operational improvement was model-dependent.** Rolling control reduced
   FlowState safety adjustment from 8.31 to 3.59 kWh and Gradient Boosting from
   14.00 to 5.92 kWh. EMA adjustment increased from 8.43 to 11.91 kWh while its
   realized throughput rose from 120.59 to 136.95 kWh, so lower cost did not
   consistently mean less corrective action.
5. **Lower quantiles exposed a safety-cost tradeoff.** Conservative q10 bounds
   reduced intervention for Gradient Boosting and the Gaussian model but sharply
   reduced battery throughput. q20 restored more use at lower cost while
   accepting more recourse.
6. **Calibration and sample count were decisive.** Nominal confidence did not
   predict realized coverage. Three EMA seeds could not distinguish q10 from
   q20, while the broad Gradient Boosting residual set over-covered and became
   expensive.
7. **The baseline-to-rolling delta is not a pure MPC effect.** The experiment
   changed several protocol and implementation details at once. The evidence
   supports the recorded totals, not a general claim that 30-minute MPC will
   always save the observed amount.

For this stable tariff, zero-PV, simple-battery experiment, once-daily control was
already economically effective and may be sufficient when inputs and execution
remain predictable. Rolling MPC is commonly motivated by resilience when PV and
load forecasts, prices, measured SOC, asset availability, or earlier actions
change during the day. Those broader disturbances were not tested here, so their
value remains a hypothesis for future evaluation rather than a conclusion from
this experiment. The observed rolling advantage was limited to small cost
differences and model-dependent recourse changes. The chance controllers were
useful diagnostics of uncertainty and risk tolerance, but no tested quantile
improved economic performance.

## Limitations and provenance

- This was one household and one 14-day retrospective test, not a field
  deployment or a new untouched model-selection period.
- The battery and tariff were synthetic, PV was zero in the battery comparison,
  and no measured battery execution was available.
- Only load uncertainty and the battery no-export rule received a chance bound.
  PV, prices, joint-horizon risk, CVaR, and battery degradation cost were not
  tested.
- Native FlowState quantiles, EMA seed spread, Gaussian marginal quantiles, and
  empirical Gradient Boosting residuals are different uncertainty concepts and
  were not equally calibrated.
- The rolling formulations were standalone CVXPY prototypes and do not prove
  identical behavior to released `naive-mpc-optim`.
- Granite PatchTST-FM and Toto 2.0 have day-ahead forecast results only. Their
  rolling and optimizer performance remains unevaluated.

The stored once-daily baseline used EMHASS 0.17.9. The rolling research code was
developed in a working tree based on
[`dd6425c1`](https://github.com/davidusb-geek/emhass/commit/dd6425c1), with Python
3.12.13, CVXPY 1.7.5, and HiGHS 1.15.1. Model-specific runners and result files
are not distributed with EMHASS because they depend on a separate research
environment. This page is a rounded retrospective record, not a turnkey
reproducibility package.

## See also

- How-to: [Rolling-horizon control with naive-mpc-optim](mpc.md)
- Explanation: [Good Practices](good_practices.md)
- Reference: [Forecasts](../forecasts.md)
- Explanation: [Advanced math model](../advanced_math_model.md)
