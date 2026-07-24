using OptimalGIV
using Test

tests = [
    "test_formula.jl",
    "test_observation_index.jl",
    "test_interface.jl",
    "test_estimates.jl",
    "test_ols_step.jl",
    "test_algorithm_equivalence.jl",
    "test_fixed_weight_mode.jl",
    "test_vcov_scope.jl",
    "test_analytic_jacobian.jl",
    "test_twostep_mc_smoke.jl",
    "test_nonfinite_guards.jl",
    # "test_with_simulations.jl"  # slow; not run in CI
]
for test in tests
    include("$test")
end
