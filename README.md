# OptimalGIV.jl

OptimalGIV estimates granular instrumental-variables models with heterogeneous
price elasticities and panel data. See the
[companion paper](https://fuzhiyu.me/TreasuryGIVPaper/Treasury_GIV_draft.pdf)
for the model and identification argument.

> [!IMPORTANT]
> **The default weighting changed from CUE to two-step.** Two-step weighting is
> more stable with generic starting values while retaining CUE-like efficiency,
> but estimates and standard errors may differ slightly from earlier results.
> Omitting `precision_weights` selects `:twostep` and emits a one-time notice
> unless `quiet = true`. Pass `precision_weights = :twostep` explicitly to
> acknowledge the new default, or `precision_weights = :cue` to use the former
> weighting scheme.

## Installation

```julia
using Pkg
Pkg.add("OptimalGIV")
```

## Minimal Example

```julia
using OptimalGIV

df = simulate_data((; M = 0.5, N = 10), Nsims = 1, seed = 1)[1]

formula = @formula(
    q + id & endog(p) ~ fe(id) + id & (η1 + η2)
)

model = giv(
    df, formula, :id, :t, :S;
    algorithm = :iv,
    precision_weights = :twostep,
    guess = fill(2.0, 10),
)

coeftable(model)
agg_coef(model)
```

The four positional arguments after the data are the formula, entity identifier,
time identifier, and nonnegative entity-size variable. Market coverage is
auto-detected when `complete_coverage` is omitted; set it explicitly only to
override that detection.

## Formula Syntax

Write the model as

```julia
@formula(q + interactions & endog(p) ~ exogenous_controls)
```

- `q` is the quantity or outcome.
- `endog(p)` marks the endogenous price variable. It belongs on the left-hand
  side; under this sign convention, a positive estimated coefficient represents
  a downward-sloping response of `q` to `p`.
- `interactions` parameterize heterogeneous elasticities. For example,
  `id & endog(p)` estimates one elasticity per entity.
- The right-hand side contains exogenous controls. Use `fe(id)` to absorb
  entity fixed effects and `pc(k)` to extract `k` common residual factors.

A homogeneous-elasticity specification is simply

```julia
@formula(q + endog(p) ~ fe(id) + controls)
```

## Precision Weighting

Use the single `precision_weights` keyword:

- `:twostep` (default) first solves with fixed precisions
  `1 / var(uqᵢ)`, where `uqᵢ` is the flow residual after absorbing fixed
  effects and controls. It then computes `1 / var(ûᵢ)` from the first-step
  residuals and solves once more with those precisions held fixed.
- `:raw_onestep` runs only the first fixed-weight solve. It is the simple,
  stable fallback.
- `:cue` recomputes residual-based precisions at each candidate estimate. Use
  it to recover the former weighting scheme.
- An entity-length vector supplies custom fixed precisions in sorted entity
  order.

`model.converged` is true for `:twostep` only when both solves converge.

## Essential Options

| Keyword | Purpose |
|---|---|
| `precision_weights` | Choose `:twostep`, `:raw_onestep`, `:cue`, or custom fixed precisions. |
| `complete_coverage` | Override automatic detection of whether the entities cover the full market. |
| `guess` | Supply a scalar, coefficient vector, or name-to-value dictionary. OLS estimates are used when omitted. |
| `algorithm` | Use `:iv` for the standard estimator. Other algorithms are available for specialized applications. |
| `exclude_pairs` | Exclude specified entity pairs from the moment conditions. |
| `save` / `save_df` | Retain residuals, fixed effects, or the processed estimation data. |
| `solver_options` | Pass additional options to NLsolve as a named tuple. |

## Main Results

The returned `GIVModel` follows the StatsAPI interface:

```julia
coef(model)
stderror(model)
vcov(model)
confint(model)
coeftable(model)
coefnames(model)
```

GIV-specific accessors are:

```julia
endog_coef(model)       # price-elasticity coefficients
exog_coef(model)        # coefficients on exogenous controls
agg_coef(model)         # aggregate or average elasticity
endog_vcov(model)
exog_vcov(model)
model.converged
model.coefdf            # coefficients organized by categorical variables
```

Categorical coefficients and entity-level results follow the natural sorted
order of their identifiers.

## Advanced Notes

- `:iv` and `:iv_twopass` support unbalanced panels. The
  `:debiased_ols` and `:scalar_search` estimators require complete market
  coverage.
- Models with `pc(k)` estimate common residual factors internally; analytical
  standard errors are not returned for these specifications.
- `build_error_function` exports the moment function and underlying matrices
  for custom diagnostics.

## Citation

Please cite:

- Gabaix, Xavier, and Ralph S. J. Koijen. “Granular Instrumental Variables.”
  *Journal of Political Economy* 132(7), 2024, 2274–2303.
- Chaudhary, Manav, Zhiyu Fu, and Haonan Zhou. “Anatomy of the Treasury Market:
  Who Moves Yields?” Available at
  [SSRN](https://ssrn.com/abstract=5021055).
