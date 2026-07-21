# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Overview

OptimalGIV.jl is a Julia package implementing Granular Instrumental Variables (GIV) estimation methods for econometric models. The package estimates models where entity-specific quantities respond to prices with heterogeneous elasticities, subject to market clearing constraints.

## Development Commands

### Package Setup
```julia
# Enter package development mode
using Pkg
Pkg.develop(path=".")  # From the GIV.jl directory

# Activate the project environment
Pkg.activate(".")

# Install dependencies
Pkg.instantiate()

# Build the package
Pkg.build("OptimalGIV")

# Precompile
Pkg.precompile()
```

### Testing
```julia
# Run all tests (excluding slow simulation tests)
Pkg.test("OptimalGIV")

# Run specific test file
include("test/test_interface.jl")
include("test/test_estimates.jl")
include("test/test_algorithm_equivalence.jl")

# Run slow simulation tests (not in CI)
include("test/test_with_simulations.jl")
```

### REPL Development
```julia
# Load package in development mode
using Revise  # Auto-reloading on file changes
using OptimalGIV

# Quick test with simulated data
df = simulate_data((; M = 0.5, N = 10), Nsims = 1, seed = 1)[1]
model = giv(df, @formula(q + id & endog(p) ~ fe(id) + id & (η1 + η2)), 
            :id, :t, :S; algorithm = :iv, precision_weights = :twostep,
            complete_coverage = true,
            guess = ones(10) * 2.0)
```

## Code Architecture

### Module Structure
- `src/OptimalGIV.jl`: Main module exporting the public API
- `src/givmodels.jl`: Defines `GIVModel` struct and result accessors
- `src/interface.jl`: Main `giv()` function and high-level API
- `src/estimation.jl`: Core algorithms (`:iv`, `:iv_twopass`, `:debiased_ols`)
- `src/scalar_search.jl`: Scalar search algorithm implementation
- `src/utils/formula.jl`: Custom formula parsing with `endog()` function
- `src/utils/ols_fe_solver.jl`: Fixed effects OLS solver
- `src/simulation.jl`: Data simulation utilities

### Algorithm Architecture

The package implements four estimation algorithms, each with different moment conditions:

1. **`:iv` (default)**: Uses E[u_i u_{S,-i}] = 0, O(N) optimized implementation; supports complete and incomplete coverage
2. **`:iv_twopass`**: Same as `:iv` but O(N²) implementation for debugging; supports complete and incomplete coverage
3. **`:debiased_ols`**: Uses E[u_i C_it p_it] = 1/ζ_St σ_i², requires complete coverage
4. **`:scalar_search`**: Searches for constant aggregate elasticity, requires a balanced panel and complete coverage

### Key Design Patterns

#### Formula Interface
The package extends StatsModels.jl with a custom `endog()` function to mark endogenous variables:
```julia
@formula(q + interactions & endog(p) ~ controls)
```
- Left side: response + interactions with endogenous variable
- Right side: exogenous controls and fixed effects

#### Data Flow
1. `preprocess_dataframe()`: Sorts by (t, id), validates panel structure
2. `estimate_giv()`: Builds matrices, runs algorithm-specific estimation
3. `build_error_function()`: Creates moment condition error function
4. Algorithm solves for parameters minimizing moment violations
5. `create_coef_dataframe()`: Organizes results by entity

#### Integration with FixedEffectModels.jl
- Parses `fe()` terms in formulas
- Uses FixedEffects.jl solvers for residualization
- Handles fixed effects absorption before main estimation

### Critical Implementation Details

#### Initial Guess Requirements
- Under the default `precision_weights = :twostep`, the default OLS guess is usually adequate; under `precision_weights = :cue`, provide a good initial guess
- Accept: scalar, vector, or Dict mapping coefficient names to values
- For `:scalar_search`: Dict with "Aggregate" key

#### Solver Configuration
- Keep dependency-specific solver settings inside `solver_options`; do not add top-level solver or PC-solver keywords

#### IV Weighting and Covariance
- `:raw_onestep` and custom entity weights use equal period weights and one fixed-weight solve
- `:twostep` first uses raw entity weights and equal period weights; it computes entity precisions and complete-coverage period weights at the first-step estimate, then freezes both for one second solve
- `:cue` recomputes every applicable weight at each candidate estimate
- Fixed-weight IV solves are exact quadratic systems under both coverage regimes
- Two-step uses the standard sandwich with the same frozen weights as its moments; the optimal covariance formula is reserved for complete-coverage CUE
- The economic multiplier is `M_t = 1 / ζS_t` for positive aggregate elasticity. The absolute-value clamp is only a finite off-domain iteration rule; final complete-coverage roots must satisfy the positive-domain check

#### Panel Data Handling
- Unbalanced panels supported for `:iv` algorithms
- Every estimator entry point requires an explicit `complete_coverage::Bool`
- Complete coverage is an economic property of the data design; in-sample market clearing validates `true` but does not select the regime
- Complete coverage is required for `:debiased_ols` and `:scalar_search`; only `:iv` and `:iv_twopass` accept incomplete coverage

#### Error Function Export
`build_error_function()` returns the raw moment condition function and matrices, enabling:
- Custom optimization with Optim.jl
- Diagnostic analysis of multiple equilibria
- Integration with other solvers

## Testing Strategy

Tests are organized by functionality:
- `test_formula.jl`: Formula parsing and `endog()` function
- `test_interface.jl`: API and input validation
- `test_estimates.jl`: Estimation accuracy against known values
- `test_algorithm_equivalence.jl`: Consistency across algorithms
- `test_with_simulations.jl`: Monte Carlo validation (slow, excluded from CI)

## Common Pitfalls

1. **Convergence Issues**: Provide good initial guesses, especially under `precision_weights = :cue`
2. **Coverage Assumptions**: Set `complete_coverage` from the data design; do not infer it from sample adding-up
3. **Missing Data**: Package doesn't handle missing values - clean data first
4. **Memory Usage**: Large panels with entity interactions can be memory-intensive
