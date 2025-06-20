#=
This script tests the bias formula when the econometrician fails to control
common factors in the data. 

The bias will be given as:
ζ̂/ζ = √(1 - R^2)

where ζ̂ is the estimated biased elasticity, and R^2 are the R^2 of common factors in prices. 
=#

using Plots
using Test, OptimalGIV, DataFrames, StatsBase, LinearAlgebra, StatsModels
using Random
Random.seed!(6)

function simulate_estimate_evaluate(Nsims;
    seed=nothing,
    formula=@formula(q + endog(p) ~ 0),
    simulation_parameters=(;),
    estimation_parameters=(;)
)
    if !isnothing(seed)
        Random.seed!(seed)
    end

    # Generate simulated datasets
    simulated_dfs = OptimalGIV.simulate_data(simulation_parameters; Nsims=Nsims, seed=seed)
    biases = Vector{Union{Missing,Float64}}(undef, Nsims)

    for (i, df) in enumerate(simulated_dfs)
        # Estimate and evaluate the model
        metrics = OptimalGIV.estimate_and_evaluate(df, formula; estimation_parameters...)

        # bias from estimate_and_evaluate: (estimated - true) where true = 1/M
        # To match the downward bias formula in terms of true elasticity
        M = simulation_parameters.M
        biases[i] = -metrics[1] * M
    end

    # Filter out missing values
    return filter(!ismissing, biases)
end

## Test 1: Symmetric case
println("Test 1: Symmetric case")
var_ushares = 0.95:-0.1:0.1
biases_symmetric = Float64[]
estparams = (;
    guess=[1.0],
    algorithm=:iv_legacy,
    quiet=true,
    return_vcov=false,
)

for v in var_ushares
    println("ushare=$(v)")
    simparams = (; T=100, N=100, ushare=v, h=0.0, σᵤcurv=0.0, K=3, M=1, missingperc=0.00, σζ=0.0)
    bias_vec = simulate_estimate_evaluate(100;
        seed=2,
        formula=@formula(q + endog(p) ~ 0),
        simulation_parameters=simparams,
        estimation_parameters=estparams
    )
    push!(biases_symmetric, mean(skipmissing(bias_vec)))
end

p1 = plot(1 .- var_ushares, biases_symmetric, label="Simulated Bias", marker=:circle)
plot!(p1, 1 .- var_ushares, 1 .- sqrt.(var_ushares), label="Approximation formula", linestyle=:dash)
title!(p1, "Symmetric Case")

## Test 2: With fat tailness
println("\nTest 2: With fat tails in size distribution")
biases_fat_tail = Float64[]

for v in var_ushares
    println("ushare=$(v)")
    simparams = (; T=100, N=10, ushare=v, h=0.37, σᵤcurv=0.38, K=1, M=1, missingperc=0.00, σζ=0)
    bias_vec = simulate_estimate_evaluate(100;
        seed=2,
        formula=@formula(q + endog(p) ~ 0),
        simulation_parameters=simparams,
        estimation_parameters=estparams
    )
    push!(biases_fat_tail, mean(skipmissing(bias_vec)))
end

p2 = plot(1 .- var_ushares, biases_fat_tail, label="Simulated Bias", marker=:circle)
plot!(p2, 1 .- var_ushares, 1 .- sqrt.(var_ushares), label="Approximation formula", linestyle=:dash)
title!(p2, "With Fat Tailness")

## Test 3: Larger N
println("\nTest 3: Larger N")
biases_large_N = Float64[]
estparams_iv = (;
    guess=[1.0],
    algorithm=:iv,
    quiet=true,
    return_vcov=false,
)

for v in var_ushares
    println("ushare=$(v)")
    simparams = (; T=100, N=100, ushare=v, h=0.2, σᵤcurv=0., K=1, M=1, missingperc=0.00, σζ=0.0)
    bias_vec = simulate_estimate_evaluate(100;
        seed=2,
        formula=@formula(q + endog(p) ~ 0),
        simulation_parameters=simparams,
        estimation_parameters=estparams_iv
    )
    push!(biases_large_N, mean(skipmissing(bias_vec)))
end

p3 = plot(1 .- var_ushares, biases_large_N, label="Simulated Bias", marker=:circle)
plot!(p3, 1 .- var_ushares, 1 .- sqrt.(var_ushares), label="Approximation formula", linestyle=:dash)
title!(p3, "Larger N")

## Test 4: Symmetric model but estimated with heterogeneous elasticity
println("\nTest 4: Symmetric model with heterogeneous elasticity estimation")
biases_hetero_est = Float64[]
estparams_hetero = (;
    guess=ones(10),
    algorithm=:iv_legacy,
    quiet=true,
    return_vcov=false,
    solver_options= (; iterations = 100, ftol=1e-4),
)

for v in var_ushares
    println("ushare=$(v)")
    simparams = (; T=100, N=10, ushare=v, h=0.2, σᵤcurv=0.0, K=1, M=1, missingperc=0.00, σζ=0.0)
    bias_vec = simulate_estimate_evaluate(100;
        seed=3,
        formula=@formula(q + id & endog(p) ~ 0),
        simulation_parameters=simparams,
        estimation_parameters=estparams_hetero
    )
    push!(biases_hetero_est, mean(trim(bias_vec, prop=0.05)))
end

p4 = plot(1 .- var_ushares, biases_hetero_est, label="Simulated Bias", marker=:circle)
plot!(p4, 1 .- var_ushares, 1 .- sqrt.(var_ushares), label="Approximation formula", linestyle=:dash)
title!(p4, "Symmetric Model with Heterogeneous Estimation")

## Test 5: With heterogeneous elasticity
println("\nTest 5: With simulated heterogeneous elasticity")
biases_hetero = Float64[]

for v in var_ushares
    println("ushare=$(v)")
    simparams = (; T=100, N=10, ushare=v, h=0.3, σᵤcurv=0, K=1, M=1, missingperc=0.00, σζ=1)
    bias_vec = simulate_estimate_evaluate(100;
        seed=3,
        formula=@formula(q + id & endog(p) ~ 0),
        simulation_parameters=simparams,
        estimation_parameters=estparams_hetero
    )
    push!(biases_hetero, mean(trim(bias_vec, prop=0.05)))
end

p5 = plot(1 .- var_ushares, biases_hetero, label="Simulated Bias", marker=:circle)
plot!(p5, 1 .- var_ushares, 1 .- sqrt.(var_ushares), label="Approximation formula", linestyle=:dash)
title!(p5, "With Simulated Heterogeneous Elasticity")

# Combine all plots
plot(p1, p2, p3, p4, p5, layout=(3, 2), size=(800, 900))
savefig("simresults/estimation_bias_simulation.png")