using OptimalGIV
using Test

tests = [
    "test_formula.jl",
    # "test_interface.jl",
    "test_estimates.jl",
    "test_algorithm_equivalence.jl"
    # "test_vcov_algorithm.jl",
    # "test_with_simulation.jl"
]
for test in tests
    include("$test")
end