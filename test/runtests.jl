using OptimalGIV
using Test

tests = [
    "test_formula.jl",
    "test_observation_index.jl",
    "test_interface.jl",
    "test_estimates.jl",
    "test_algorithm_equivalence.jl",
    # "test_with_simulations.jl"  # slow; not run in CI
]
for test in tests
    include("$test")
end