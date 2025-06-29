using Test
using OptimalGIV
using DataFrames, Random, Statistics
using OptimalGIV: evaluation_metrics, simulate_data, estimate_and_evaluate

# Monte Carlo simulation parameters
Random.seed!(123)

simparams = (
    N=20,         # Number of entities
    T=100,        # Number of time periods  
    K=2,          # Number of common factors
    M=0.5,        # Aggregate elasticity
    σζ=0.0,       # Homogeneous elasticity (no heterogeneity)
    σp=2.0,       # Price volatility
    σᵤcurv=0.1,
    h=0.2,        # Excess HHI
    ushare=0.6,   # Share of idiosyncratic shocks
    missingperc=0.0  # No missing data
)

# Number of simulations
Nsims = 100

# Generate all simulated datasets
println("Generating $Nsims simulated datasets...")
simulated_dfs = simulate_data(simparams, Nsims=Nsims, seed=101)

# Storage for results
results_observed = DataFrame(ζ_bias=Float64[], ζ_se=Float64[], β_bias=Float64[], β_se=Float64[])
results_pc = DataFrame(ζ_bias=Float64[], ζ_se=Float64[], β_bias=Float64[], β_se=Float64[])
results_nopc = DataFrame(ζ_bias=Float64[], ζ_se=Float64[], β_bias=Float64[], β_se=Float64[])

# Run simulations
println("\nRunning estimations on $Nsims simulations...")
for (i, df) in enumerate(simulated_dfs)
    # Progress indicator
    if i % 10 == 0
        println("  Processing simulation $i/$Nsims...")
    end

    # Convert id to string for consistency
    df.id = string.(df.id)
    sort!(df, [:t, :id])

    # Method 1: Estimation with observed factors (oracle)
    metrics_observed = OptimalGIV.estimate_and_evaluate(
        df,
        @formula(q + endog(p) ~ 0 + id & (η1 + η2)),
        guess=[2.0],  # Single elasticity since homogeneous
        algorithm=:iv_twopass,
        quiet=true
    )
    push!(results_observed, metrics_observed; promote=true)

    # Method 2: Estimation with PC extraction
    metrics_pc = OptimalGIV.estimate_and_evaluate(
        df,
        @formula(q + endog(p) ~ pc(2) + 0),  # Extract 2 PCs
        guess=[2.0],  # Single elasticity since homogeneous
        algorithm=:iv_twopass,
        quiet=true,
        tol=2e-4,)
    push!(results_pc, metrics_pc; promote=true)

    # Method 3: Estimation without factors (baseline)
    metrics_nopc = OptimalGIV.estimate_and_evaluate(
        df,
        @formula(q + endog(p) ~ 0),  # No factors
        guess=[2.0],  # Single elasticity since homogeneous
        algorithm=:iv_twopass,
        quiet=true
    )
    push!(results_nopc, metrics_nopc; promote=true)
end

# Helper function to compute coverage
ci_coverage(bias, se) = mean(abs.(bias ./ se) .< 1.96)

# Summarize results
println("\n" * "="^60)
println("MONTE CARLO RESULTS ($Nsims simulations)")
println("="^60)

# Method 1: Observed factors (oracle)
obs_clean = dropmissing(results_observed)
n_obs = nrow(obs_clean)
println("\n1. OBSERVED FACTORS (Oracle with true η1, η2):")
println("   Successful runs: $n_obs/$Nsims")
if n_obs > 0
    println("   Elasticity bias: $(round(mean(obs_clean.ζ_bias), digits=4)) " *
            "(median: $(round(median(obs_clean.ζ_bias), digits=4)))")
    println("   Elasticity SE: $(round(mean(obs_clean.ζ_se), digits=4))")
    println("   Coverage: $(round(ci_coverage(obs_clean.ζ_bias, obs_clean.ζ_se), digits=3))")
    if !all(isnan.(obs_clean.β_bias))
        println("   Factor loading bias: $(round(mean(obs_clean.β_bias), digits=4))")
        println("   Factor loading SE: $(round(mean(obs_clean.β_se), digits=4))")
    end
end

# Method 2: PC extraction
pc_clean = dropmissing(results_pc)
n_pc = nrow(pc_clean)
println("\n2. PC EXTRACTION (pc(2)):")
println("   Successful runs: $n_pc/$Nsims")
if n_pc > 0
    println("   Elasticity bias: $(round(mean(pc_clean.ζ_bias), digits=4)) " *
            "(median: $(round(median(pc_clean.ζ_bias), digits=4)))")
    println("   Elasticity SE: $(round(mean(pc_clean.ζ_se), digits=4))")
    println("   Coverage: $(round(ci_coverage(pc_clean.ζ_bias, pc_clean.ζ_se), digits=3))")
end

# Method 3: No factors (baseline)
nopc_clean = dropmissing(results_nopc)
n_nopc = nrow(nopc_clean)
println("\n3. NO FACTORS (Baseline):")
println("   Successful runs: $n_nopc/$Nsims")
if n_nopc > 0
    println("   Elasticity bias: $(round(mean(nopc_clean.ζ_bias), digits=4)) " *
            "(median: $(round(median(nopc_clean.ζ_bias), digits=4)))")
    println("   Elasticity SE: $(round(mean(nopc_clean.ζ_se), digits=4))")
    println("   Coverage: $(round(ci_coverage(nopc_clean.ζ_bias, nopc_clean.ζ_se), digits=3))")
end

println("\n" * "="^60)

# Simple assertions to verify performance
println("\nPerformance checks:")
println("  Oracle convergence rate: $(n_obs/$Nsims)")
println("  PC convergence rate: $(n_pc/$Nsims)")
println("  Baseline convergence rate: $(n_nopc/$Nsims)")

if n_obs > 0
    println("  Oracle bias magnitude: $(abs(mean(obs_clean.ζ_bias)))")
end

if n_pc > 0 && n_nopc > 0
    println("  PC vs Baseline bias reduction: $(abs(mean(nopc_clean.ζ_bias)) / abs(mean(pc_clean.ζ_bias)))")
end
