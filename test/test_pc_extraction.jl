using Test, OptimalGIV, DataFrames, Random, Statistics, LinearAlgebra
using HeteroPCA: HeteroPCAModel, DeflatedHeteroPCA
using HeteroPCA: matrix_distance, sinθ_distance
using OptimalGIV: vector_to_matrix, matrix_to_vector, extract_pcs_from_residuals, create_observation_index
using OptimalGIV: simulate_data

@testset "PC Extraction Utilities Tests" begin
    Random.seed!(123)
    
    # Create test data
    df = DataFrame(
        id = repeat(1:3, outer=5),
        t = repeat(1:5, inner=3),
        S = repeat([0.3, 0.4, 0.3], outer=5),
        value = randn(15)
    )
    obs_index = create_observation_index(df, :id, :t)
    
    @testset "Core PC extraction functionality" begin
        residuals = randn(15)
        n_pcs = 2
        
        # Test vector/matrix conversion roundtrip
        residual_matrix = vector_to_matrix(residuals, obs_index)
        recovered_residuals = matrix_to_vector(residual_matrix, obs_index)
        @test size(residual_matrix) == (3, 5)  # N×T
        @test residuals ≈ recovered_residuals
        
        # Test specific value mapping
        @test residual_matrix[1, 1] ≈ residuals[1]  # entity 1, time 1
        @test residual_matrix[3, 5] ≈ residuals[15]  # entity 3, time 5
        
        # Test PC extraction
        factors, loadings_matrix, pca_model, updated_residuals = extract_pcs_from_residuals(residuals, obs_index, n_pcs)
        
        # Check output dimensions and types
        @test size(factors) == (2, 5)  # k×T
        @test size(loadings_matrix) == (3, 2)  # N×k
        @test typeof(pca_model) <: HeteroPCAModel
        @test length(updated_residuals) == length(residuals)
        
        # Check finite values
        @test all(isfinite.(factors))
        @test all(isfinite.(loadings_matrix))
        @test all(isfinite.(updated_residuals))
        
        # Check that residuals changed and variance reduced
        @test updated_residuals != residuals
        @test var(updated_residuals) < var(residuals)  # PCs should capture variation
    end
    
    @testset "Edge cases and special scenarios" begin
        residuals = randn(15)
        
        # Test with n_pcs = 0 (no PC extraction)
        factors_0, loadings_0, model_0, updated_0 = extract_pcs_from_residuals(residuals, obs_index, 0)
        @test size(factors_0, 1) == 0  # 0×T matrix
        @test size(loadings_0, 2) == 0  # N×0 matrix
        @test updated_0 ≈ residuals  # Residuals unchanged when no PCs extracted
        
        # Test with n_pcs = max possible
        factors_max, loadings_max, _, _ = extract_pcs_from_residuals(residuals, obs_index, 3)
        @test size(factors_max, 1) <= 3
        @test size(loadings_max, 2) <= 3
        
        # Test with very small residuals (numerical stability)
        small_residuals = fill(1e-10, 15)
        _, _, _, updated_small = extract_pcs_from_residuals(small_residuals, obs_index, 1)
        @test all(isfinite.(updated_small))
        
        # Test with unbalanced panel
        df_unbalanced = df[1:12, :]  # Remove last 3 observations
        obs_index_unb = create_observation_index(df_unbalanced, :id, :t)
        
        factors_unb, loadings_unb, _, updated_unb = extract_pcs_from_residuals(randn(12), obs_index_unb, 1)
        @test size(factors_unb, 1) == 1
        @test size(loadings_unb, 2) == 1
        @test length(updated_unb) == 12
    end
end

# ========================================
# Simulation-Based PC Extraction Tests
# ========================================

"""
    generate_giv_factor_data(N, T, K; missing_prob=0.0, seed=42)

Generate realistic GIV simulation data with known factor structure.

Returns:
- `demand_shocks`: Vector of demand shocks containing the factor structure  
- `true_factors`: K×T matrix of true factors (η)
- `true_loadings`: N×K matrix of true loadings (Λ)
- `true_residuals`: Vector of true idiosyncratic residuals (u)
- `obs_index`: Observation index for the panel
"""
function generate_giv_factor_data(N::Int, T::Int, K::Int;
    missing_prob::Float64=0.0,
    seed::Int=42)
    # Use existing GIV simulation infrastructure
    simparams = (N=N, T=T, K=K, σζ=0.1, missingperc=missing_prob)
    simdata = simulate_data(simparams; Nsims=1, seed=seed)[1]

    # Extract factor and loading column names
    factor_cols = [Symbol("η$i") for i in 1:K]
    loading_cols = [Symbol("λ$i") for i in 1:K]

    # Extract true factors - they're repeated for each entity, so take unique values
    factors_df = unique(simdata[!, [:t; factor_cols]], :t)
    sort!(factors_df, :t)
    factors_matrix = Matrix(factors_df[!, factor_cols])  # T×K
    true_factors = factors_matrix'  # K×T (transpose)

    # Extract true loadings - they're repeated for each time period, so take unique values  
    loadings_df = unique(simdata[!, [:id; loading_cols]], :id)
    sort!(loadings_df, :id)
    true_loadings = Matrix(loadings_df[!, loading_cols])  # N×K

    # Construct demand shocks: q + ζ * p (this contains the factor structure)
    # Work directly with the vectors to handle missing data properly
    demand_shocks_vector = simdata.q + simdata.ζ .* simdata.p

    # Create observation index
    df = DataFrame(
        id=simdata.id,
        t=simdata.t,
        value=demand_shocks_vector,
        u=simdata.u  # Include true residuals
    )

    # Remove missing observations
    df = dropmissing(df, [:value, :u])

    obs_index = create_observation_index(df, :id, :t)
    demand_shocks = df.value
    true_residuals = df.u

    return demand_shocks, true_factors, true_loadings, true_residuals, obs_index
end

"""
    compute_factor_recovery_metrics(est_factors, est_loadings, true_factors, true_loadings, 
                                  updated_residuals, true_residuals)

Compute distance metrics for factor recovery and residual comparison.
"""
function compute_factor_recovery_metrics(est_factors, est_loadings, true_factors, true_loadings,
    updated_residuals, true_residuals)
    # 1. Subspace distance - measures angle between factor spaces (scale-invariant)
    subspace_distance = sinθ_distance(est_loadings, true_loadings)

    # 2. Normalized factor distance - factors normalized to unit Frobenius norm
    true_factors_norm = true_factors ./ norm(true_factors, 2)
    est_factors_norm = est_factors ./ norm(est_factors, 2)
    factor_distance_norm = matrix_distance(est_factors_norm', true_factors_norm', align=true, relative=false)

    # 3. Residual comparison - distance between extracted residuals and true residuals
    residual_distance = norm(updated_residuals - true_residuals) / norm(true_residuals)

    return (
        subspace_distance=subspace_distance,
        factor_distance_norm=factor_distance_norm,
        residual_distance=residual_distance
    )
end

@testset "PC Factor Recovery Tests" begin
    # Test parameters: Large panel (N=100, T=1000) with K=2 true factors
    # This tests whether PC extraction can recover the true factor structure
    # from GIV residuals containing both factors and idiosyncratic noise
    N, T, K = 100, 1000, 2
    n_sims = 30

    @testset "DeflatedHeteroPCA Factor Recovery" begin
        # This test validates that our PC extraction correctly recovers known factors
        # from simulated GIV data where we know the true factor structure
        
        # Storage for results
        deflated_metrics = []

        for sim in 1:n_sims
            # Generate realistic GIV data with known factor structure
            # demand_shocks = true_factors * true_loadings' + true_residuals
            demand_shocks, true_factors, true_loadings, true_residuals, obs_index = generate_giv_factor_data(N, T, K; seed=42 + sim)

            # Extract factors using DeflatedHeteroPCA algorithm
            # This should recover factors close to the true ones
            factors_def, loadings_def, model_def, updated_residuals = extract_pcs_from_residuals(
                demand_shocks, obs_index, K;
                algorithm=DeflatedHeteroPCA()
            )

            # Compute recovery quality metrics
            metrics_def = compute_factor_recovery_metrics(factors_def, loadings_def, true_factors, true_loadings,
                updated_residuals, true_residuals)

            push!(deflated_metrics, metrics_def)
        end

        # Aggregate results across simulations
        def_subspace_dist = mean([m.subspace_distance for m in deflated_metrics])
        def_factor_norm = mean([m.factor_distance_norm for m in deflated_metrics])
        def_residual_dist = mean([m.residual_distance for m in deflated_metrics])

        # Validate recovery quality:
        # - Subspace distance: measures angle between estimated and true factor spaces (0 = perfect)
        #   For K=2, theoretical range is [0, √2] ≈ [0, 1.41]. Good recovery should be < 0.5
        @test def_subspace_dist < 0.5
        
        # - Factor distance: normalized Frobenius distance between factor matrices (0 = perfect)
        #   After optimal rotation alignment. Good recovery should be < 0.6
        @test def_factor_norm < 0.6
        
        # - Residual distance: relative distance between true and extracted residuals
        #   Measures how well we separated factors from idiosyncratic components
        @test def_residual_dist < 0.2

        # Print results for inspection
        println("DeflatedHeteroPCA Factor Recovery Quality:")
        println("  Subspace distance: $(round(def_subspace_dist, digits=4))")
        println("  Factor distance (normalized): $(round(def_factor_norm, digits=4))")
        println("  Residual distance: $(round(def_residual_dist, digits=4))")
    end

    @testset "Missing Data Robustness" begin
        # This test validates that PC extraction remains accurate even with missing data
        # HeteroPCA is designed to handle missing values through iterative imputation
        
        missing_probs = [0.0, 0.1, 0.2]  # Test with 0%, 10%, and 20% missing data
        results = []

        for missing_prob in missing_probs
            sim_results = []

            for sim in 1:10  # Fewer simulations for missing data test
                # Generate data with specified missing percentage
                demand_shocks, true_factors, true_loadings, true_residuals, obs_index = generate_giv_factor_data(
                    N, T, K; missing_prob=missing_prob, seed=100 + sim
                )

                # Extract factors - DeflatedHeteroPCA should handle missing data internally
                factors, loadings_matrix, _, updated_residuals = extract_pcs_from_residuals(
                    demand_shocks, obs_index, K;
                    algorithm=DeflatedHeteroPCA()
                )

                # Compute subspace recovery quality
                metrics = compute_factor_recovery_metrics(factors, loadings_matrix, true_factors, true_loadings,
                    updated_residuals, true_residuals)
                push!(sim_results, metrics.subspace_distance)
            end

            avg_subspace_dist = mean(sim_results)
            push!(results, avg_subspace_dist)

            # Verify reasonable recovery even with missing data
            # We allow slightly worse performance but still expect decent recovery
            @test avg_subspace_dist < 0.5
        end

        println("Missing data robustness - average subspace distances:")
        println("  Missing %: $(missing_probs)")
        println("  Distances: $(round.(results, digits=4))")
        println("  (Lower is better; <0.5 indicates good recovery)")
    end
end