# News

## v0.3.0 (unreleased)

### Breaking changes

1. **New default estimator: feasible two-step GMM (`precision_weights=:twostep`) replaces the continuous updating estimator of earlier versions (`:cue`) for `:iv` and `:iv_twopass`.** The moment conditions weight each entity by its precision, `1/var(u_i)`, and, under complete coverage, each period by the multiplier `M_t = 1/ζS_t`. Both weights depend on ζ, and `:cue` recomputes them at every candidate, which made the solver fragile from generic starting values. Step 1 instead solves with precisions from `q` residualized on the controls and equal period weights. Step 2 recomputes both weights at the step-1 estimate, freezes them, and solves once more. Without `pc(k)`, each solve is an exact quadratic system. Pass `precision_weights=:cue` to `giv` or `build_error_function` for the earlier point estimates or moment map.
2. **`complete_coverage` is required.** Coverage is a property of the data design, and in-sample adding-up cannot establish it. Declaring `true` on data that fail adding-up now errors.
3. **Standard errors change for incomplete-coverage and `exclude_pairs` fits.** Incomplete coverage uses an empirical sandwich, which stays consistent when the variance model fails, and the covariance now drops excluded pairs to match the moments. No option restores the earlier values.
4. **Roots with nonpositive or near-zero aggregate elasticity report `converged=false`** under complete coverage, so the numerical clamp cannot pass off a non-economic root.
5. **Invalid combinations now error.** `exclude_pairs` with `:debiased_ols` or `:scalar_search` was silently ignored, and those algorithms ran on `complete_coverage=false` when `quiet=true`.
6. **Standard errors of the exogenous coefficients change.** Each entity's OLS variance uses only the periods it is observed, which corrects the common-`T` scaling on unbalanced panels: an entity's own coefficients were understated by a factor of `n_i/T`, and pooled coefficients could move either way. It also uses `n_i - k_i` degrees of freedom, with `k_i` the entity's leverage, which rescales balanced panels by `(n_i - 1)/(n_i - k_i)`. Effects absorbed by `fe()` are not counted in `k_i`. Point estimates, the elasticity covariance, and `residual_variance` do not change.

### New

- `vcov` selects `:auto`, `:sandwich`, or `:optimal`, because the efficient information formula relies on time-constant entity variances and cross-entity-independent second moments.
- `precision_weights` also accepts `:raw_onestep` or custom entity weights.
- Fixed-weight IV uses an analytic Jacobian, and nonfinite starting moments raise a named `NonfiniteMomentError`.
