using Test
using OptimalGIV
using OptimalGIV: evaluation_metrics
using DataFrames, CSV, CategoricalArrays, Statistics
using HeteroPCA: DeflatedHeteroPCA

# ========================================
# Helper Functions
# ========================================

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
            println("[$simparamstr] [$estimate_label] Estimating simulation $i of $(length(simdata_files))")
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

# ========================================
# Setup and Load Simulations
# ========================================

# First ensure simulations exist
if !isfile("$(@__DIR__)/../simulations/simparamstr.csv")
    @info "simparamstr.csv not found, running simulate_data.jl to generate simulations"
    include("simulate_data.jl")
end

# Load simulation parameters and convert to dictionary
simparamdf = CSV.read("$(@__DIR__)/../simulations/simparamstr.csv", DataFrame)
simparams_dict = Dict(row.simulated_model => row.simparamstr for row in eachrow(simparamdf))

# Initialize results collection
all_results = DataFrame()

# ========================================
# Standard Model Tests
# ========================================

@testset "Standard Model Specifications" begin
    # Define test configurations
    standard_test_configs = [
        # (sim_label, formula, estimate_label, guess, test_params)
        ("baseline", @formula(q + id & endog(p) ~ 0 + id & (η1 + η2)), "entity_specific", nothing,
            (bias_tol=0.05, β_bias_tol=0.1, coverage_range=(0.925, 0.975), min_success=80)),
        ("10% missing", @formula(q + id & endog(p) ~ 0 + id & (η1 + η2)), "entity_specific", nothing,
            (bias_tol=0.05, β_bias_tol=0.1, coverage_range=(0.925, 0.975), min_success=80)),
        ("sparse panel", @formula(q + endog(p) ~ 0 + fe(id) & (η1 + η2)), "fixed_effects", [2.0],
            (bias_tol=0.5, β_bias_tol=nothing, coverage_range=(0.85, 0.975), min_success=nothing)),
        ("large panel", @formula(q + endog(p) ~ 0 + id & (η1 + η2)), "entity_specific", [2.0],
            (bias_tol=0.01, β_bias_tol=0.02, coverage_range=(0.92, 0.98), min_success=nothing))
    ]

    for (sim_label, formula, estimate_label, guess, test_params) in standard_test_configs
        @testset "$estimate_label: $sim_label" begin
            metrics = run_simulation_estimation(
                simparams_dict[sim_label],
                formula,
                Nsims=400,
                estimate_label=estimate_label,
                guess=guess
            )

            performance = summarize_metrics(metrics)
            performance.simulation .= sim_label
            performance.estimate_label .= estimate_label
            performance.simparamstr .= simparams_dict[sim_label]
            append!(all_results, performance)

            # Apply test assertions based on configuration
            if !isnothing(test_params.min_success)
                @test performance.n_successful[1] > test_params.min_success
            end
            @test abs(performance.ζ_bias_mean[1]) < test_params.bias_tol
            if !isnothing(test_params.β_bias_tol)
                @test abs(performance.β_bias_mean[1]) < test_params.β_bias_tol
            end
            @test test_params.coverage_range[1] < performance.ζ_covered[1] <= test_params.coverage_range[2]
            if !isnothing(test_params.β_bias_tol)
                @test test_params.coverage_range[1] < performance.β_covered[1] <= test_params.coverage_range[2]
            end
        end
    end
end

# ========================================
# PC Extraction Tests
# ========================================

@testset "PC Extraction Methods" begin
    # Define PC test configurations
    pc_test_configs = [
        ("homogeneous_large", "Large panel (N=40, T=400)"),
        ("homogeneous_large_missing", "Large panel with 20% missing")
    ]

    # Define estimation methods for PC extraction
    pc_estimation_methods = [
        ("deflated_heteropca", @formula(q + endog(p) ~ 0 + pc(2)),
            Dict(:algorithm => :iv, :return_vcov => false, :tol => 1e-6,
                :pca_option => (; impute_method=:zero, demean=false, maxiter=100,
                    algorithm=DeflatedHeteroPCA(t_block=10), abstol=1e-8))),
        ("no_factors", @formula(q + endog(p) ~ 0),
            Dict(:algorithm => :iv, :return_vcov => false, :tol => 1e-6)),
        ("known_factors", @formula(q + endog(p) ~ 0 + fe(id) & (η1 + η2)),
            Dict(:algorithm => :iv, :return_vcov => false, :tol => 1e-6))
    ]

    for (sim_key, sim_desc) in pc_test_configs
        @testset "$sim_desc" begin
            for (method_label, formula, method_kwargs) in pc_estimation_methods
                @testset "$method_label" begin
                    metrics = run_simulation_estimation(
                        simparams_dict[sim_key],
                        formula,
                        Nsims=400,
                        estimate_label=method_label,
                        guess=[1.0],
                        quiet=true;
                        method_kwargs...
                    )

                    performance = summarize_metrics(metrics)
                    performance.simulation .= sim_key
                    performance.estimate_label .= method_label
                    performance.simparamstr .= simparams_dict[sim_key]
                    append!(all_results, performance)

                    # Method-specific assertions
                    if method_label == "no_factors"
                        @test abs(performance.ζ_bias_mean[1]) < 1.1
                    else
                        @test abs(performance.ζ_bias_mean[1]) < 0.1
                    end
                    # Note: no_factors expected to have bias

                    # Optional: print debug info if needed
                    if get(ENV, "VERBOSE_TESTS", "false") == "true"
                        println("$method_label - Success rate: $(performance.n_successful[1])/100")
                        println("$method_label - Elasticity bias: $(round(performance.ζ_bias_mean[1], digits=4))")
                    end
                end
            end
        end
    end
end

# ========================================
# Save Results
# ========================================

# Create results directory if it doesn't exist
mkpath("simresults")

# Save all results
CSV.write("simresults/simulation_performance.csv", all_results)

# # Also save PC-specific results for backward compatibility
# pc_results = filter(row -> row.estimate_label in ["deflated_heteropca", "no_factors", "known_factors"], all_results)
# CSV.write("simresults/internal_pc_simulation_performance.csv", pc_results)