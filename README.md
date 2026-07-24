# OptimalGIV.jl

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://fuzhiyu.github.io/OptimalGIV.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://fuzhiyu.github.io/OptimalGIV.jl/dev/) -->
[![Build Status](https://github.com/fuzhiyu/OptimalGIV.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/fuzhiyu/OptimalGIV.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Build Status](https://app.travis-ci.com/fuzhiyu/OptimalGIV.jl.svg?branch=main)](https://app.travis-ci.com/fuzhiyu/OptimalGIV.jl)
[![Coverage](https://codecov.io/gh/fuzhiyu/OptimalGIV.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/fuzhiyu/OptimalGIV.jl)

Estimate models using granular instrumental variables (GIV), with efficient weighting schemes.

The core algorithms are tested using simulations, but the documentation is under development and minor features may contain bugs. Feature requests and bug reports are welcome. For details of the model and estimator, see [the companion paper](https://fuzhiyu.me/TreasuryGIVPaper/Treasury_GIV_draft.pdf).

For Python users, a Python wrapper can be found [here](https://github.com/FuZhiyu/optimalgiv).

### Model Specification

The GIV model estimated by this package follows the specification:

```math
\begin{aligned}
\left.\begin{array}{c}
\begin{array}{cl}
q_{i,t} & =-p_{t}\times\mathbf{C}_{i,t}'\boldsymbol{\zeta}+\mathbf{X}_{i,t}'\boldsymbol{\beta}+u_{i,t},\\
0 & =\sum_{i}S_{i,t}q_{i,t}
\end{array}\end{array}\right\} \implies & p_{t}=\frac{1}{\mathbf{C}_{S,t}'\boldsymbol{\zeta}}\left[\mathbf{X}_{S,t}'\boldsymbol{\beta}+u_{S,t}\right],
\end{aligned}
```

where $q_{i,t}$ and $p_t$ are endogenous, $S$ is the entity-size variable, and the subscript $S$ denotes an $S$-weighted sum.

The model is estimated with the moment condition $\mathbb E[u_{i,t}u_{j,t}] = 0$ for non-excluded entity pairs.

Unbalanced panels are allowed. The caller must declare whether the sampled entities cover the full market with `complete_coverage=true` or `false`. In-sample adding-up, $\sum_i S_{i,t}q_{i,t}=0$, is required when complete coverage is declared, but it cannot establish that omitted entities do not exist. The package therefore validates `true` rather than inferring coverage from adding-up.

> [!NOTE]
> `precision_weights` defaults to `:twostep` (feasible two-step GMM); the
> earlier default was continuously updated GMM (`:cue`). Pass
> `precision_weights = :cue` explicitly to reproduce that estimator.


## Installation

```julia
using Pkg
Pkg.add("OptimalGIV")
```


## Usage

### Basic Example

```julia
using OptimalGIV, DataFrames

# Using simulated panel data
df = simulate_data((; M = 0.5, N = 10), Nsims = 1, seed = 1)[1]

# Estimate the model
model = giv(df,
    @formula(q + id & endog(p) ~ fe(id) + id & (η1 + η2)),
    :id, :t, :S;
    algorithm = :iv,
    complete_coverage = true,
    precision_weights = :twostep,
    save = :all, # fixed effects will also be saved in the coefdf
    guess = ones(10) * 2.0
)

# View results
println(model)
#                       GIVModel (Aggregate coef: 2.21)
# ───────────────────────────────────────────────────────────────────────────
#             Estimate  Std. Error   t-stat  Pr(>|t|)    Lower 95%  Upper 95%
# ───────────────────────────────────────────────────────────────────────────
# id: 1 & p    3.58315    1.39997   2.55945    0.0106   0.835899      6.33039
# id: 10 & p   2.85081    0.567497  5.02347    <1e-06   1.73717       3.96444
# id: 2 & p    2.07155    0.592432  3.49668    0.0005   0.908981      3.23411
# id: 3 & p    1.17017    0.593346  1.97216    0.0489   0.00581219    2.33453
# id: 4 & p    1.17624    0.608863  1.93187    0.0537  -0.0185676     2.37105
# ...
```

### Formula Specification

The formula interface generally follows the [`StatsModel.jl`](https://github.com/JuliaStats/StatsModels.jl) and [`FixedEffectModels.jl`](https://github.com/FixedEffects/FixedEffectModels.jl), with small twists to indicate the endogenous variables using `endog` function:


```julia
@formula(q + interactions & endog(p) ~ exog_controls + pc(k))
```

- `q`: Response variable (e.g., quantity)
- `endog(p)`: Endogenous variable (e.g., price). Endogenous variables appear on the left-hand side; hence positive coefficients indicate negative responses of `q` on `p` (downward-sloping demand curve).
- `interactions`: Exogenous variables to parameterize heterogeneous elasticities (e.g., entity identifiers or characteristics)
- `exog_controls`: Exogenous control variables. Fixed effects as in `FixedEffectModels.jl` are allowed.
- `pc(k)`: Principal component extraction with `k` factors (optional). When specified, `k` common factors are extracted from residuals using HeteroPCA.jl


#### Examples of formulas:

```julia
# Homogeneous elasticity with entity specific loadings (estimated) and fixed effects (absorbed)
@formula(q + endog(p) ~ id & η + fe(id))

# Heterogeneous elasticity by entity
@formula(q + id & endog(p) ~ id & η + fe(id))

# Multiple interactions
@formula(q + id & endog(p) + category & endog(p) ~ fe(id) & η1 + η2)

@formula(q + id & endog(p) ~ 0 + id & η)

# With PC extraction (2 factors)
@formula(q + endog(p) ~ 0 + pc(2))

# exogneous controls with PC extraction
@formula(q + endog(p) ~ fe(id) & η1 + pc(3))
```

### Key Function: `giv()`

```julia
giv(df, formula, id, t, weight; kwargs...)
```

#### Arguments:
- `df`: DataFrame with panel data (must be balanced for some algorithms)
- `formula`: Model specification using `@formula`
- `id`: Symbol for entity identifier column
- `t`: Symbol for time identifier column
- `weight`: Symbol for entity weights/sizes (e.g., market shares)

#### Keyword Arguments:
- `algorithm`: `:iv` (default), `:debiased_ols`, `:scalar_search`, or `:iv_twopass`
- `guess`: Initial parameter guess (vector, number, or Dict)
- `exclude_pairs`: Dictionary specifying entity pairs to exclude from moment conditions.
  Example: `Dict(1 => [2, 3], 4 => [5])` excludes pairs (1,2), (1,3), and (4,5)
- `quiet`: Suppress warnings and information messages if true (default: false)
- `save`: Save additional information - `:none` (default), `:residuals`, `:fe`, or `:all`
- `save_df`: If true, the full estimation DataFrame (including residuals, coefficients, and fixed-effects columns when requested) is stored in the returned model. When PC extraction is used, PC factors and loadings are also included.
- `complete_coverage`: Required Boolean declaring whether the sampled entities cover the full market. A `true` declaration is validated by the in-sample adding-up condition. Adding-up does not select the coverage regime.
- `precision_weights`: IV weighting mode: `:twostep` (default), `:raw_onestep`, `:cue`, or a custom entity-length vector. See [Estimator Weighting](#estimator-weighting).
- `vcov`: Covariance route: `:auto` (default), `:sandwich`, or `:optimal`. See [Covariance Routing](#covariance-routing).
- `return_vcov`: Calculate variance-covariance matrix (default: true, automatically disabled when PC extraction is used)
- `contrasts`: Contrasts specification for categorical variables (following StatsModels.jl). Untested. Use with cautions.
- `tol`: Convergence tolerance (default: 1e-6)
- `iterations`: Maximum iterations (default: 100)
- `solver_options`: Options passed to `NLsolve.jl` as a named tuple
- `pca_option`: Options for HeteroPCA.jl PC extraction (default: `(; impute_method=:zero, demean=false, maxiter=1000, algorithm=DeflatedHeteroPCA(t_block=10))`)

### Working with Results

The `giv()` function returns a `GIVModel` object with various fields and methods:

```julia
# Basic statistics
coef(model)              # All coefficient estimates (endogenous + exogenous)
endog_coef(model)        # Coefficients on endogenous terms (ζ)
exog_coef(model)         # Coefficients on exogenous control variables (β)

agg_coef(model)           # Aggregate (when complete_coverage=false, report average instead) elasticity for each t

vcov(model)              # Full variance-covariance matrix
endog_vcov(model)        # Variance-covariance of endog_coef
exog_vcov(model)         # Variance-covariance of exog_coef

stderror(model)          # Standard errors (same order as coef)
confint(model)           # Confidence intervals
coeftable(model)         # Formatted coefficient table

# Model information
nobs(model)              # Number of observations
dof_residual(model)      # Residual degrees of freedom
formula(model)           # Model formula

# Access specific fields
coefnames(model)         # Names of all coefficients
endog_coefnames(model)   # Names of endogenous-term coefficients
exog_coefnames(model)    # Names of exogenous-term coefficients

model.coefdf             # DataFrame with entity-specific coefficients (see below)
model.converged          # Convergence status
model.n_pcs              # Number of principal components extracted
model.pc_factors         # PC factors (k×T matrix, or nothing if n_pcs=0)
model.pc_loadings        # PC loadings (N×k matrix, or nothing if n_pcs=0)
model.pc_model           # HeteroPCA model object (or nothing if n_pcs=0)
```

#### Ordering of Results

All categorical variables (including `id` variable) in the model follow their **natural sort order**:
- For numeric categories: sorted numerically (e.g., 3, 5, 10, 20)
- For string categories: sorted alphabetically (e.g., "firm_A", "firm_B", "firm_C")

This applies to:
- Coefficient vectors when categorical variables are used in interactions
- The residual variance vector (`model.residual_variance`), which follows the entity ID order
- The factor loading matrix
- The `model.coefdf` DataFrame, which organizes results by categorical variables

Additionally:
- Any DataFrame returned by the model (e.g., when `save_df = true`) is sorted by `[t, id]`

#### Entity-specific Coefficients DataFrame (`coefdf`)

The `model.coefdf` field provides a convenient way to access and report coefficients organized by categorical variables (e.g., by sector, entity, or other groupings). This DataFrame contains:

- All categorical variable values used in the model (e.g., entity IDs, sectors)
- Estimated coefficients for each term in the formula, stored in columns named `<term>_coef`
- Fixed effect estimates (if `save = :fe` or `save = :all` was specified)

Example:
```julia
# Using the estimated model above as an example
# Access the coefficient DataFrame
first(model.coefdf, 5)
# 5×4 DataFrame
#  Row │ id      id & p_coef  id & η1_coef  id & η2_coef  fe_id
#      │ String  Float64      Float64       Float64       Float64
# ─────┼────────────────────────────────────────────────────────────
#    1 │ 1           3.58315     6.67398         1.95733  0.550752
#    2 │ 10          2.85081    -0.0851448       1.68483  0.0775327
#    3 │ 2           2.07155     2.87146         2.91998  0.738134
#    4 │ 3           1.17017     3.55465         4.05976  0.36872
#    5 │ 4           1.17624     0.542161        1.91074  0.470043
```

## Estimator Weighting

The `precision_weights` modes below apply to `algorithm=:iv` and its reference
implementation, `:iv_twopass`. Here $\widetilde{\zeta}$ denotes the first-step
estimate and $M_t$ is the complete-coverage period multiplier.

| Mode | Entity weights | Period weights | IV moment system | Covariance (`vcov=:auto`) |
|---|---|---|---|---|
| `:raw_onestep` | Fix $1/\operatorname{var}(uq_i)$ before solving | Equal | Exact quadratic | Sandwich with the same fixed weights (not eligible for the information formula) |
| Custom vector | Fix the supplied entity weights before solving | Equal | Exact quadratic | Sandwich with the same fixed weights (not eligible for the information formula) |
| `:twostep` (default) | Step 1 uses raw weights; step 2 fixes $1/\operatorname{var}(\widehat u_i(\widetilde{\zeta}))$ | Step 1 is equal; complete-coverage step 2 fixes $M_t(\widetilde{\zeta})$ | Two exact-quadratic solves | Information formula when eligible (converged, complete coverage, in-domain); sandwich with the frozen step-2 weights otherwise |
| `:cue` | Recompute residual-based weights at every candidate | Recompute $M_t$ at every candidate under complete coverage; none otherwise | Nonlinear | Information formula when eligible (converged, complete coverage, in-domain); sandwich otherwise (including incomplete coverage) |

Feasible two-step GMM therefore has exactly two solves. Step 1 uses raw entity
weights and equal period weights to obtain $\widetilde{\zeta}$. The estimator
then computes both applicable weight families at $\widetilde{\zeta}$ and holds
them fixed during the single step-2 solve for $\widehat{\zeta}_2$. It does not
update weights within step 2.

The paper defines $M_t=1/\zeta_{S,t}$ on the maintained domain
$\zeta_{S,t}>0$. To keep trial evaluations finite during nonlinear iteration,
the package evaluates a normalized version of
`1 / clamp(abs(ζS_t), sqrt(eps), Inf)` off that domain. This numerical extension
does not change the estimand: a reported complete-coverage root with
nonpositive or near-zero aggregate elasticity is marked non-converged.

## Covariance Routing

The `vcov` keyword selects the covariance route. `:auto` (default) reports the
model-implied information-formula covariance for an eligible fit — converged,
complete-coverage, in-domain `:twostep` or `:cue` — and the masked empirical
sandwich otherwise (see the table above). `:sandwich` forces the sandwich,
built from exactly the entity and period weights frozen in the estimating
moments and the same non-excluded entity pairs used by those moments, on any
supported route. `:optimal` requires the information formula and errors when
its restrictions fail. The information formula is efficient under a
maintained second-moment model (time-constant entity variances,
cross-entity-independent second moments); the sandwich stays consistent when
that model fails, so the robust alternative is one keyword away.

The sandwich's Jacobian follows the orientation of the pairwise IV estimating
equations themselves — it is the empirical bread/meat estimator, distinct
from the model-implied information matrix `:optimal` reports.

## Algorithms

The package implements four algorithms for GIV estimation. Coverage support is:

| Algorithm | Complete coverage | Incomplete coverage |
|---|---:|---:|
| `:iv` | Yes | Yes |
| `:iv_twopass` | Yes | Yes |
| `:debiased_ols` | Yes | No |
| `:scalar_search` | Yes | No |

Every call must specify the coverage regime. `:debiased_ols` and
`:scalar_search` are separate full-market estimators; the fixed-quadratic and
`precision_weights` descriptions above concern the two IV implementations.

### 1. `:iv` (Instrumental Variables)
The most flexible algorithm using the moment condition E[u_i u_{S,-i}] = 0. This is the default and recommended algorithm for most applications. It uses an efficient O(N) implementation. It allows for:

- Exclude certain pairs $E[u_i u_j] = 0$ from the moment conditions;
- Flexible elasticity specifications;
- Unbalanced panel with incomplete market coverage;
- PC extraction: Supports internal factor extraction using `pc(k)` in formulas

### 2. `:iv_twopass`
Numerically identical to `:iv` but uses a more straightforward O(N²) implementation with two passes over entity pairs. This is useful for:
- Debugging purposes
- When the O(N) optimization in `:iv` might cause numerical issues
- When there are many pairs to be excluded, which will slow down the algorithm in :iv.
- Understanding the computational flow of the moment conditions
- PC extraction: Supports internal factor extraction using `pc(k)` in formulas

### 3. `:debiased_ols`
Uses the moment condition E[u_i C_it p_it] = 1/ζ_St σ_i². Requires the sampled entities to cover the full market and the adding-up constraint to hold. More efficient when applicable but more restrictive.
- PC extraction: Not supported with this algorithm

### 4. `:scalar_search`
Efficient algorithm when the aggregate elasticity is constant across time. Searches for a scalar aggregate elasticity value. Useful for diagnostics or forming initial guesses. Requires:
- Balanced panel data
- Constant weights across time
- Complete market coverage
- PC extraction: Not supported with this algorithm

## Internal PCA

Internal PC extractions are supported. With internal PCs, the moment conditions become $\mathbb E[u_{i,t}u_{j,t}] = \Lambda \Lambda'$, where $\Lambda$ is the factor loadings estimated internally using [HeteroPCA.jl](https://github.com/FuZhiyu/HeteroPCA.jl) from $u_{i,t}(z) \equiv q_{i,t} + p_{t}\times\mathbf{C}_{i,t}'\boldsymbol{z}$ at each guess of $z$. However, following caveats apply:

- With internal PC extraction, the weighting scheme is no longer optimal as it does not consider the covariance in the moment conditions due to common factor estimation. The standard error formula also no longer applies and hence was not returned. One can consider bootstrapping for statistical inference;

- In small samples, the exactly root solving the moment condition may not exist, and users may want to use an minimizer to minimize the error instead.

- A model with fully flexible elasticity specification and fully flexible internal factor loadings is not theoretically identifiable. Hence, one needs to assume certain level of homogeneity to estimate factors internally.

## Initial Guesses

When `guess` is omitted, the package uses OLS coefficients. This is usually adequate for the default fixed-weight first step; `precision_weights=:cue` is more sensitive and may require a well-chosen starting value.

Initial parameter guesses can be provided in several formats:

```julia
# Single number (for homogeneous elasticity)
guess = 1.0

# Vector (in order of coefficients)
guess = [1.0, 2.0, 3.0]

# Dictionary with parameter names (use get_coefnames to check the coefficient labels)
guess = Dict("id: 1 & p" => 1.0, "id: 2 & p" => 2.0)

# For scalar_search algorithm
guess = Dict("Aggregate" => 2.5)
```

To see the order of coefficients or get the coefficient labels, one can use the helper function:
```julia
response, endog_name, endog_coefnames, exog_coefnames, slope_terms =
    get_coefnames(df, formula)
```

## Debugging and Customization

### Error Functions and Low-level Matrices

The `build_error_function` API allows you to extract the error function and low-level matrices used in GIV estimation. The error function takes the vector of elasticity guess (scalar for the scalar-search algorithm) and returns the errors of the moment condtitions. This is particularly useful for:

- Using alternative optimization solvers (e.g., Optim.jl)
- Diagnosing convergence issues
- Performing custom analyses

```julia
# Export the error function and components
err_func, components = build_error_function(df,
    @formula(q + endog(p) ~ fe(id) + id & (η1 + η2)),
    :id, :t, :S;
    algorithm = :iv,
    complete_coverage = true,
    precision_weights = :raw_onestep,
)

# The returned components depend on the algorithm:
# For :iv, the components include uq, uCp, C, S, and obs_index, where uq and
# uCp are q and the endogenous-price interactions residualized against the
# right-hand side.
# For :scalar_search: (uqmat=uqmat, p=p, S_vec=S_vec, coefmapping=coefmapping)

# Use with custom optimization
using Optim
initial_guess = [1.0]
result = optimize(x->sum(err_func(x).^2), initial_guess, LBFGS())
```

For homogeneous-elasticity models or the scalar-search algorithm, you can use interval search to analyze the structure of the error function:

```julia
# Plot the error function over an interval
using Plots
ζ_range = 0.5:0.01:3.0
plot(x-> err_func([x])[1], ζ_range, xlabel="Elasticity", ylabel="Error",
     title="Error Function Structure")

# Find all roots in an interval
using Roots
roots = find_zeros(ζ -> err_func([ζ])[1], 0.1, 5.0)
```

The error function represents the moment conditions:
- For `:iv`: E[u_i u_{S,-i}] = 0
- For `:debiased_ols`: E[u_i C_it p_it] - σ_i²/ζ_St = 0
- For `:scalar_search`: Searches for aggregate elasticity ζ_S

Access to these low-level functions enables advanced users to implement custom estimation procedures or diagnostic tools.

`build_error_function(...; precision_weights=:twostep)` exports the raw
first-step map only; it does not run two estimator solves. Use `giv` to estimate
the feasible two-step model.

### Simulation

The package includes utilities for Monte Carlo simulations using the `simulate_data` function:

```julia
# Generate simulated panel datasets
simulated_dfs = simulate_data(
    (; N = 20,      # Number of entities
       T = 50,      # Time periods
       K = 3,       # Number of factors
       M = 0.7,     # Aggregate elasticity
       σζ = 0.5),   # Elasticity dispersion
    Nsims = 1,      # Number of simulations
    seed = 123      # Random seed
)

# Use the first dataset
df = simulated_dfs[1]
```

#### Simulation Parameters

The `simulate_data` function accepts a NamedTuple with the following parameters:

- `N`: Number of entities (default: 10)
- `T`: Number of time periods (default: 100)
- `K`: Number of common factors (default: 2)
- `M`: Aggregate price elasticity (default: 0.5)
- `σζ`: Standard deviation of entity elasticities (default: 1.0)
- `σp`: Price volatility to target (default: 2.0)
- `h`: Excess HHI for size distribution (default: 0.2)
- `ushare`: Share of price variation explained by idiosyncratic shocks (default: 0.2 if K>0)
- `σᵤcurv`: Curvature for size-dependent volatility (default: 0.1)
- `ν`: Degrees of freedom for t-distribution (default: Inf = Normal)
- `missingperc`: Percentage of missing values (default: 0.0)

The generated data follows:
- `q_it = u_it + Λ_i * η_t - ζ_i * p_t`
- `p_t = M * Σ_i S_i * (u_it + Λ_i * η_t)`
- Entity sizes follow a power law distribution

## Limitations

- **PC extraction limitations**: Only `:iv` and `:iv_twopass` algorithms support internal PC extraction. The `:debiased_ols` and `:scalar_search` algorithms do not support PC extraction.
- **Variance-covariance matrix**: When PC extraction is used (`pc(k)` in formula), the variance-covariance matrix calculation is automatically disabled as it is not correct. One should consider bootstrapping instead.
- Time fixed effects are not supported directly, but one can use a single factor `pc(1)` instead;
- Some algorithms require balanced panels
- The `:debiased_ols` and `:scalar_search` algorithms require complete market coverage; only `:iv` and `:iv_twopass` support incomplete coverage
- Coverage must be supplied explicitly; an in-sample adding-up check cannot establish that the dataset includes the whole market

## To-do List

- Support for standard GIV
- Interface with RegressionTables.jl

## References

Please cite:

- Gabaix, Xavier, and Ralph S.J. Koijen. Granular Instrumental Variables. Journal of Political Economy, 132(7), 2024, pp. 2274–2303.
- Chaudhary, Manav, Zhiyu Fu, and Haonan Zhou. Anatomy of the Treasury Market: Who Moves Yields? Available at SSRN: https://ssrn.com/abstract=5021055
