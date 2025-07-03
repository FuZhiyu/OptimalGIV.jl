using Test, OptimalGIV, Random, Statistics, DataFrames
using OptimalGIV: simulate_data, evaluation_metrics
using OptimalGIV.HeteroPCA: DeflatedHeteroPCA, StandardHeteroPCA, DiagonalDeletion
using Roots
# Simple simulation parameters for homogeneous elasticities
simparams = (
    N=40,         # Number of entities
    T=400,         # Number of time periods  
    K=2,          # Number of common factors
    M=0.5,        # Aggregate elasticity (1/M = 2.0 true elasticity)
    σζ=0.0,       # Homogeneous elasticity (no heterogeneity)
    σp=2,       # Price volatility
    σᵤcurv=0.1,   # Size-dependent shock curvature
    h=0.2,        # Excess HHI
    ushare=0.5,   # Share of idiosyncratic shocks
    missingperc=0.0  # No missing data
)
##
df_list = simulate_data(simparams, seed=rand(1:1000), Nsims=100)
# giv(df_list[1], @formula(q + endog(p) ~ 0 + pc(2)), :id, :t, :S; algorithm=:iv_twopass, pca_option=(; impute_method=:zero, demean=false, maxiter=100, algorithm=StandardHeteroPCA(), suppress_warnings=true, abstol=1e-8), guess=[1.0])
giv(df_list[1], @formula(q + endog(p) ~ 0 + pc(2)), :id, :t, :S; algorithm=:iv, pca_option=(; impute_method=:zero, demean=false, maxiter=100, algorithm=StandardHeteroPCA(), suppress_warnings=true, abstol=1e-8), guess=[1.0])

df = df_list[1]
##

err, elem = build_error_function(df, @formula(q + endog(p) ~ 0 + pc(2)), :id, :t, :S; algorithm=:iv_twopass, pca_option=(; impute_method=:zero, demean=true, maxiter=10000, algorithm=DeflatedHeteroPCA(t_block=1000), abstol=1e-8))
plot(x -> err([x])[1], 1.5:0.3:5.0)

##
using StatsBase
using DataFramesMeta, FixedEffectModels
function estimate_standard_giv(df, k; pca_option=(; impute_method=:zero, demean=false, maxiter=100, algorithm=StandardHeteroPCA(), suppress_warnings=true, abstol=1e-8))
    df = @chain df begin
        groupby(:id)
        @combine(:precision = 1 ./ var(skipmissing(:q)))
        @transform!(:precision = :precision ./ sum(:precision))
        innerjoin(df, on=:id)
    end
    m = reg(df, @formula(q ~ fe(t)), weights=:precision, save=:residuals)
    obsindex = OptimalGIV.create_observation_index(df, :id, :t)
    _, _, _, u = OptimalGIV.extract_pcs_from_residuals(m.residuals, obsindex, k; pca_option...)
    df.u = u
    giv = @chain df begin
        groupby(:id)
        @transform!(:precision = 1 ./ var(skipmissing(:u)))
        @transform!(:precision = :precision ./ sum(:precision))
        groupby(:t)
        @combine(:z = sum(skipmissing(:S .* :u)), :qE = mean(:q, weights(:precision)), :p = first(:p))
    end
    reg(giv, @formula(qE ~ (p ~ z)))
end
-estimate_standard_giv(df, 2).coef[2]






# err, elem = build_error_function(df, @formula(q + endog(p) ~ 0 + fe(id) & (η1 + η2)), :id, :t, :S; algorithm=:iv_twopass, pca_option=(; impute_method=:zero, demean=false, maxiter=10000, algorithm=DeflatedHeteroPCA(t_block=50), abstol=1e-8))
plot(x -> err([x])[1], 1.5:0.3:5.0)


df.u_wrong = df.q + df.p * 2.0
using FixedEffectModels
reg(df, @formula(u_wrong ~ fe(t)))

m = giv(df, @formula(q + endog(p) ~ 0 + pc(2)), :id, :t, :S; algorithm=:iv_twopass, pca_option=(; impute_method=:zero, demean=false, maxiter=100, algorithm=StandardHeteroPCA(), suppress_warnings=true, abstol=1e-8, maxiter=1000), guess=[1.1], tol=1e-8, quiet=false)

using Plots

err(coef(m))[1]

## Finding roots for error function across all datasets using IntervalRootFinding

println("\n" * "="^60)
println("Finding roots of error function for all datasets")
println("="^60)

# Function to find roots using Roots.jl find_zeros
function find_roots_simple(err_fn, search_interval=(0.01, 10.0); tolerance=1e-3)
    a, b = search_interval

    # Create a scalar version of error function
    scalar_err(ζ) = err_fn([ζ])[1]

    # Use find_zeros to find all roots in the interval
    # It automatically finds multiple roots
    roots_found = find_zeros(scalar_err, a, b)

    # Filter out roots that are not actually zeros (due to numerical error)
    true_roots = Float64[]
    for root in roots_found
        if abs(scalar_err(root)) < tolerance
            push!(true_roots, root)
        end
    end

    # Return at most 2 roots
    return true_roots[1:min(2, length(true_roots))]
end

# Analyze all datasets
results = []
for (i, df) in enumerate(df_list)
    # Build error function
    err, elem = build_error_function(
        df,
        @formula(q + endog(p) ~ 0 + pc(2)),
        pca_option=(; impute_method=:zero, demean=false, maxiter=100, algorithm=DeflatedHeteroPCA(t_block=10), suppress_warnings=true, abstol=1e-6),
        :id, :t, :S;
        algorithm=:iv_twopass
    )

    # Find roots using find_zeros
    roots = find_roots_simple(err, (0.01, 10.0), tolerance=1e-3)

    push!(results, (
        dataset=i,
        roots=roots,
        n_roots=length(roots),
        true_elasticity=1 / simparams.M  # Should be 2.0
    ))

    # Print results for this dataset
    if !isempty(roots)
        println("\nDataset $i: Found $(length(roots)) root(s)")
        for (j, root) in enumerate(roots)
            println("  Root $j: ζ = $(round(root, digits=4)) (error = $(round(err([root])[1], digits=8)))")
        end
    end
end

# Summary statistics
println("\n" * "="^60)
println("SUMMARY STATISTICS")
println("="^60)

n_total = length(df_list)
n_with_roots = sum(r.n_roots > 0 for r in results)
n_with_one_root = sum(r.n_roots == 1 for r in results)
n_with_two_roots = sum(r.n_roots == 2 for r in results)
n_with_more = sum(r.n_roots > 2 for r in results)

println("Total datasets: $n_total")
println("Datasets with roots: $n_with_roots ($(round(100*n_with_roots/n_total, digits=1))%)")
println("  - With 1 root: $n_with_one_root")
println("  - With 2 roots: $n_with_two_roots")
println("  - With >2 roots: $n_with_more")

# Extract all roots
all_roots = Float64[]
for r in results
    append!(all_roots, r.roots)
end

if !isempty(all_roots)
    println("\nRoot distribution:")
    println("  Number of roots found: $(length(all_roots))")
    println("  Mean: $(round(mean(all_roots), digits=4))")
    println("  Median: $(round(median(all_roots), digits=4))")
    println("  Std: $(round(std(all_roots), digits=4))")
    println("  Range: [$(round(minimum(all_roots), digits=4)), $(round(maximum(all_roots), digits=4))]")
    println("  True elasticity: $(1/simparams.M)")

    # Check how many are close to true value
    true_val = 1 / simparams.M
    close_to_true = sum(abs(r - true_val) < 0.1 for r in all_roots)
    println("  Roots close to true value (±0.1): $close_to_true ($(round(100*close_to_true/length(all_roots), digits=1))%)")
end

# Analyze datasets with exactly two roots
two_root_datasets = filter(r -> r.n_roots == 2, results)
# if !isempty(two_root_datasets)
println("\n" * "="^60)
println("ANALYSIS OF DATASETS WITH TWO ROOTS")
println("="^60)

# Extract smaller and larger roots
smaller_roots = [r.roots[1] for r in two_root_datasets]
larger_roots = [r.roots[2] for r in two_root_datasets]

println("\nNumber of datasets with exactly 2 roots: $(length(two_root_datasets))")

println("\nSmaller roots statistics:")
println("  Mean: $(round(mean(smaller_roots), digits=4))")
println("  Median: $(round(median(smaller_roots), digits=4))")
println("  Std: $(round(std(smaller_roots), digits=4))")
println("  Range: [$(round(minimum(smaller_roots), digits=4)), $(round(maximum(smaller_roots), digits=4))]")

println("\nLarger roots statistics:")
println("  Mean: $(round(mean(larger_roots), digits=4))")
println("  Median: $(round(median(larger_roots), digits=4))")
println("  Std: $(round(std(larger_roots), digits=4))")
println("  Range: [$(round(minimum(larger_roots), digits=4)), $(round(maximum(larger_roots), digits=4))]")

println("\nTrue elasticity: $(1/simparams.M)")

# Plot distributions
using Plots
mean(smaller_roots)
# Create histogram of smaller and larger roots
p1 = histogram(smaller_roots,
    bins=20,
    label="Smaller roots",
    xlabel="ζ",
    ylabel="Count",
    title="Distribution of Smaller Roots (Two-Root Cases)",
    color=:blue,
    alpha=0.7,
    xlims=(0, 4))
vline!(p1, [1 / simparams.M], label="True value", color=:red, linewidth=2)

p2 = histogram!(larger_roots,
    bins=20,
    label="Larger roots",
    xlabel="ζ",
    ylabel="Count",
    title="Distribution of Larger Roots (Two-Root Cases)",
    color=:green,
    alpha=0.7,
    xlims=(0, 4))
vline!(p2, [1 / simparams.M], label="True value", color=:red, linewidth=2)

# Combined plot
p_combined = plot(p1, p2, layout=(2, 1), size=(800, 600))
display(p_combined)

# Scatter plot showing relationship between smaller and larger roots
p_scatter = scatter(smaller_roots, larger_roots,
    xlabel="Smaller Root",
    ylabel="Larger Root",
    title="Relationship between Smaller and Larger Roots",
    label="Data points",
    markersize=4,
    alpha=0.6)
plot!(p_scatter, [0, 10], [0, 10], label="y=x", color=:black, linestyle=:dash)
vline!(p_scatter, [1 / simparams.M], label="True value", color=:red, linewidth=1, alpha=0.5)
hline!(p_scatter, [1 / simparams.M], label="", color=:red, linewidth=1, alpha=0.5)
display(p_scatter)

# Check which root is closer to true value
true_val = 1 / simparams.M
smaller_closer = sum(abs.(smaller_roots .- true_val) .< abs.(larger_roots .- true_val))
larger_closer = length(two_root_datasets) - smaller_closer

println("\nWhich root is closer to true value?")
println("  Smaller root closer: $smaller_closer ($(round(100*smaller_closer/length(two_root_datasets), digits=1))%)")
println("  Larger root closer: $larger_closer ($(round(100*larger_closer/length(two_root_datasets), digits=1))%)")
# end

# Plot error function for a few examples with two roots
println("\nPlotting error functions for first 3 datasets with exactly two roots...")
two_root_examples = two_root_datasets[1:min(3, length(two_root_datasets))]

for r in two_root_examples
    df = df_list[r.dataset]
    err, _ = build_error_function(df, @formula(q + endog(p) ~ 0 + pc(2)), :id, :t, :S;
        algorithm=:iv,
        pca_option=(; impute_method=:zero, demean=false, maxiter=100, algorithm=DeflatedHeteroPCA(t_block=10), suppress_warnings=true, abstol=1e-6))

    p = plot(x -> err([x])[1], 0.01, 10.0,
        label="Error function",
        xlabel="ζ",
        ylabel="Error",
        title="Dataset $(r.dataset) - Error function with 2 roots",
        legend=:topright,
        linewidth=2)

    # Mark the roots with different colors
    plot!(p, [r.roots[1]], [0], seriestype=:scatter,
        label="Smaller root: $(round(r.roots[1], digits=3))",
        markersize=8, color=:blue)
    plot!(p, [r.roots[2]], [0], seriestype=:scatter,
        label="Larger root: $(round(r.roots[2], digits=3))",
        markersize=8, color=:green)

    # Add true value
    vline!(p, [1 / simparams.M], label="True value", color=:red, linewidth=2, linestyle=:dash)

    # Add horizontal line at y=0
    hline!(p, [0], color=:black, linestyle=:dash, label="")

    display(p)
end