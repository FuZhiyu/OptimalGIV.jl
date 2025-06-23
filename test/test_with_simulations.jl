using Test
using OptimalGIV
using DataFrames, CSV, CategoricalArrays, Statistics
using OptimalGIV: evaluation_metrics

# Helper functions
function preprocess_simulation_data(df::DataFrame; min_obs_per_id=5)
    """
    Preprocess simulation data by dropping IDs with less than min_obs_per_id non-missing observations
    """
    # Group by ID and filter groups with sufficient non-missing observations
    df.id = CategoricalArray(df.id)
    gdf = groupby(df, :id)
    return DataFrame(filter(g -> count(!ismissing, g.q) >= min_obs_per_id, gdf))
end

function estimate_simulated_model(df::DataFrame, formula;
    guess=nothing,
    save=:none,
    quiet=true,
    solver_options=(; ftol=1e-4, iterations=100,),
    kwargs...)

    df.id = CategoricalArray(df.id)
    if isnothing(guess)
        guess = unique(df, :id).ζ
    end
    model = giv(df, formula, :id, :t, :S;
        guess=guess, save=save, quiet=quiet,
        solver_options=solver_options, kwargs...)
    return model
end

function estimate_and_evaluate(df::DataFrame, formula; kwargs...)
    df.id = CategoricalArray(df.id)
    model = estimate_simulated_model(df, formula; kwargs...)
    return evaluation_metrics(model, df)
end

function run_simulation_estimation(simparamstr::String, formula;
    Nsims=100,
    estimate_label="standard",
    verbose=true,
    min_obs_per_id=5,
    kwargs...)
    # Use simparamstr as the folder name directly
    simpath = joinpath("$(@__DIR__)/../simulations", simparamstr)
    simdata_files = readdir(simpath)
    simdata_files = joinpath.(simpath, filter(x -> occursin("simdata_", x), simdata_files))
    simdata_files = simdata_files[1:min(Nsims, length(simdata_files))]

    metricdf = DataFrame(ζ_bias=Float64[], ζ_se=Float64[],
        β_bias=Float64[], β_se=Float64[])

    for i in 1:length(simdata_files)
        if verbose && i % 10 == 0
            println("[$(simparamstr)] [$estimate_label] Estimating simulation $i of $(length(simdata_files))")
        end

        try
            # Load and preprocess data
            df = CSV.read(simdata_files[i], DataFrame)
            df_processed = preprocess_simulation_data(df; min_obs_per_id=min_obs_per_id)

            # Skip if no valid IDs remain
            if nrow(df_processed) == 0
                @warn "No valid IDs in simulation $(simdata_files[i]) after preprocessing"
                continue
            end

            est_metric = estimate_and_evaluate(df_processed, formula; kwargs...)
            push!(metricdf, est_metric; promote=true)
        catch e
            @warn "Error in simulation $(simdata_files[i]): $e"
        end
    end

    metricdf.simparamstr .= simparamstr
    metricdf.estimate_label .= estimate_label
    return metricdf
end

function summarize_metrics(metricdf)
    sdf = dropmissing(metricdf)
    if nrow(sdf) == 0
        return DataFrame()
    end

    ci_covered = (bias, se) -> mean(abs.(bias ./ se) .< 1.96)
    return combine(sdf,
        [:ζ_bias, :ζ_se, :β_bias, :β_se] .=> mean .=> (x -> x * "_mean"),
        [:ζ_bias, :ζ_se, :β_bias, :β_se] .=> median .=> (x -> x * "_median"),
        [:ζ_bias, :ζ_se] => ci_covered => "ζ_covered",
        [:β_bias, :β_se] => ci_covered => "β_covered",
        [:ζ_bias, :β_bias] .=> std => (x -> x * "_std"),
        nrow => :n_successful)
end

# Main test sets
# First ensure simulations exist
if !isfile("$(@__DIR__)/../simulations/simparamstr.csv")
    @info "simparamstr.csv not found, running simulate_data.jl to generate simulations"
    include("simulate_data.jl")
end

# Load simulation parameters and convert to dictionary
simparamdf = CSV.read("$(@__DIR__)/../simulations/simparamstr.csv", DataFrame)
simparams_dict = Dict(row.simulated_model => row.simparamstr for row in eachrow(simparamdf))

# Test different model specifications with bias and coverage together
# @testset "Model Estimation and Coverage" begin
results = DataFrame()
# Standard specification tests
#================== Baseline tests ==================#
standard_sims = ["baseline", "10% missing"]
@testset "Standard: $sim_label" for sim_label in standard_sims
    # Standard specification with entity-specific elasticities
    metrics = run_simulation_estimation(
        simparams_dict[sim_label],
        @formula(q + id & endog(p) ~ 0 + id & (η1 + η2)),
        Nsims=400,
        estimate_label="entity_specific"
    )

    performance_metrics = summarize_metrics(metrics)
    performance_metrics.simulation .= sim_label
    append!(results, performance_metrics)
    # Basic checks
    @test performance_metrics.n_successful[1] > 80  # At least 80% success rate
    @test abs(performance_metrics.ζ_bias_mean[1]) < 0.05  # Reasonable bias
    @test abs(performance_metrics.β_bias_mean[1]) < 0.1  # Reasonable bias
    @test 0.925 < performance_metrics.ζ_covered[1] <= 0.975  # Coverage should be   close to 95%
    @test 0.925 < performance_metrics.β_covered[1] <= 0.975  # Coverage should be close to 95%
end

#==================  sparse panel with higher tolerance ==================#
@testset "Sparse panel with higher tolerance" begin
    sim_label = "sparse panel"
    # Fixed effects specification
    metrics = run_simulation_estimation(
        simparams_dict[sim_label],
        @formula(q + endog(p) ~ 0 + fe(id) & (η1 + η2)),
        Nsims=400,  # Fewer simulations for speed
        estimate_label="standard",
        guess=[2.0]  # Single elasticity for uniform case
    )

    performance_metrics = summarize_metrics(metrics)
    performance_metrics.simulation .= sim_label
    append!(results, performance_metrics)
    # For uniform elasticity (σζ=0.0), check that bias is small
    @test abs(performance_metrics.ζ_bias_mean[1]) < 0.5
    @test 0.85 < performance_metrics.ζ_covered[1] <= 0.975
end

#================== sparse and long panel ==================#
@testset "Sparse and long panel" begin
    sim_label = "large panel"
    # Fixed effects specification
    metrics = run_simulation_estimation(
        simparams_dict[sim_label],
        @formula(q + endog(p) ~ 0 + id & (η1 + η2)),
        Nsims=400,  # Fewer simulations for speed
        estimate_label="standard",
        guess=[2.0]  # Single elasticity for uniform case
    )

    performance_metrics = summarize_metrics(metrics)
    performance_metrics.simulation .= sim_label
    append!(results, performance_metrics)
    # For uniform elasticity (σζ=0.0), check that bias is small
    @test abs(performance_metrics.ζ_bias_mean[1]) < 0.01
    @test abs(performance_metrics.β_bias_mean[1]) < 0.02
    @test 0.925 < performance_metrics.ζ_covered[1] <= 0.975
    @test 0.92 < performance_metrics.β_covered[1] <= 0.98
end

CSV.write("simresults/simulation_performance.csv", results)