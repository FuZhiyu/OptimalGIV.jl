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
    
    # Create observation index
    obs_index = create_observation_index(df, :id, :t)
    
    @testset "vector_to_matrix and matrix_to_vector" begin
        # Test residuals vector
        residuals = randn(15)
        
        # Convert to matrix and back
        residual_matrix = vector_to_matrix(residuals, obs_index)
        recovered_residuals = matrix_to_vector(residual_matrix, obs_index)
        
        # Check dimensions
        @test size(residual_matrix) == (3, 5)  # N×T
        @test length(recovered_residuals) == 15  # same as original
        
        # Check round-trip consistency
        @test residuals ≈ recovered_residuals
        
        # Check specific values
        # First observation: entity 1, time 1 -> should be at [1,1]
        @test residual_matrix[1, 1] ≈ residuals[1]
        # Last observation: entity 3, time 5 -> should be at [3,5] 
        @test residual_matrix[3, 5] ≈ residuals[15]
    end
    
    @testset "extract_pcs_from_residuals" begin
        # Test residuals vector
        residuals = randn(15)
        n_pcs = 2
        
        # Extract PCs (now always returns updated residuals)
        factors, loadings_matrix, pca_model, updated_residuals = extract_pcs_from_residuals(residuals, obs_index, n_pcs)
        
        # Check dimensions
        @test size(factors) == (2, 5)  # k×T
        @test size(loadings_matrix) == (3, 2)  # N×k
        @test typeof(pca_model) <: HeteroPCAModel
        @test length(updated_residuals) == length(residuals)
        
        # Check that factors and loadings are numeric
        @test all(isfinite.(factors))
        @test all(isfinite.(loadings_matrix))
        @test all(isfinite.(updated_residuals))
    end
    
    @testset "residual updating properties" begin
        # Test residuals vector  
        residuals = randn(15)
        n_pcs = 2
        
        # Extract PCs and update residuals
        _, _, _, updated_residuals = extract_pcs_from_residuals(residuals, obs_index, n_pcs)
        
        # Check dimensions
        @test length(updated_residuals) == length(residuals)
        
        # Check that residuals changed (PC components removed)
        @test updated_residuals != residuals
        
        # Check that updated residuals are finite
        @test all(isfinite.(updated_residuals))
        
        # Check that variance is reduced (PCs capture some variation)
        original_var = var(residuals)
        updated_var = var(updated_residuals)
        @test updated_var <= original_var  # Should be less or equal
    end
    
    @testset "PC extraction with unbalanced panel" begin
        # Create unbalanced panel by removing some observations
        df_unbalanced = df[1:12, :]  # Remove last 3 observations
        obs_index_unb = create_observation_index(df_unbalanced, :id, :t)
        
        residuals_unb = randn(12)
        n_pcs = 1
        
        # Should still work with unbalanced panel
        @test_nowarn begin
            factors, loadings_matrix, _, updated_residuals = extract_pcs_from_residuals(residuals_unb, obs_index_unb, n_pcs)
            @test size(factors, 1) == 1  # k=1
            @test size(loadings_matrix, 2) == 1  # k=1
            @test length(updated_residuals) == 12
        end
    end
    
    @testset "Edge cases" begin
        residuals = randn(15)
        
        # Test with n_pcs = 0 (should return empty matrices)
        factors_0, loadings_0, model_0, updated_0 = extract_pcs_from_residuals(residuals, obs_index, 0)
        @test size(factors_0, 1) == 0  # 0×T matrix
        @test size(loadings_0, 2) == 0  # N×0 matrix
        @test length(updated_0) == length(residuals)
        
        # Test with n_pcs larger than min(N,T)
        # With N=3, T=5, max meaningful PCs is min(3,5)=3
        @test_nowarn begin
            factors, loadings_matrix, _, _ = extract_pcs_from_residuals(residuals, obs_index, 3)
            @test size(factors, 1) <= 3
            @test size(loadings_matrix, 2) <= 3
        end
        
        # Test with very small residuals (near zero)
        small_residuals = fill(1e-10, 15)
        @test_nowarn begin
            _, _, _, updated = extract_pcs_from_residuals(small_residuals, obs_index, 1)
            @test all(isfinite.(updated))
        end
    end
    
    @testset "Consistency of PC extraction" begin
        residuals = randn(15)
        n_pcs = 2
        
        # Extract PCs multiple times - should give same results
        factors1, loadings1, model1, updated_residuals1 = extract_pcs_from_residuals(residuals, obs_index, n_pcs)
        factors2, loadings2, model2, updated_residuals2 = extract_pcs_from_residuals(residuals, obs_index, n_pcs)
        
        # Check that results are consistent
        @test factors1 ≈ factors2 atol=1e-12
        @test loadings1 ≈ loadings2 atol=1e-12
        @test updated_residuals1 ≈ updated_residuals2 atol=1e-12
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
    # Test parameters
    N, T, K = 100, 1000, 2
    n_sims = 30

    @testset "DeflatedHeteroPCA Factor Recovery" begin
        # Storage for results
        deflated_metrics = []

        for sim in 1:n_sims
            # Generate realistic GIV data with known factor structure
            demand_shocks, true_factors, true_loadings, true_residuals, obs_index = generate_giv_factor_data(N, T, K; seed=42 + sim)

            # Test DeflatedHeteroPCA  
            factors_def, loadings_def, model_def, updated_residuals = extract_pcs_from_residuals(
                demand_shocks, obs_index, K;
                algorithm=DeflatedHeteroPCA()
            )

            # Compute metrics
            metrics_def = compute_factor_recovery_metrics(factors_def, loadings_def, true_factors, true_loadings,
                updated_residuals, true_residuals)

            push!(deflated_metrics, metrics_def)
        end

        # Aggregate results
        def_subspace_dist = mean([m.subspace_distance for m in deflated_metrics])
        def_factor_norm = mean([m.factor_distance_norm for m in deflated_metrics])
        def_residual_dist = mean([m.residual_distance for m in deflated_metrics])

        # Tests: Verify that factor recovery works (distances are finite and better than worst case)
        # For sinθ_distance with K=2: range is [0, √2] ≈ [0, 1.41]
        @test def_subspace_dist < 0.5   # Good subspace recovery
        @test def_factor_norm < 0.6     # Good factor recovery
        @test def_residual_dist < 0.2   # Reasonable residual recovery

        # Print results for inspection
        println("DeflatedHeteroPCA:")
        println("  Subspace distance: $(round(def_subspace_dist, digits=4))")
        println("  Factor distance (normalized): $(round(def_factor_norm, digits=4))")
        println("  Residual distance: $(round(def_residual_dist, digits=4))")
    end

    @testset "Missing Data Robustness" begin
        # Test with missing data
        missing_probs = [0.0, 0.1, 0.2]
        results = []

        for missing_prob in missing_probs
            sim_results = []

            for sim in 1:10  # Fewer simulations for missing data test
                demand_shocks, true_factors, true_loadings, true_residuals, obs_index = generate_giv_factor_data(
                    N, T, K; missing_prob=missing_prob, seed=100 + sim
                )

                # Extract factors with DeflatedHeteroPCA (should handle missing data well)
                factors, loadings_matrix, _, updated_residuals = extract_pcs_from_residuals(
                    demand_shocks, obs_index, K;
                    algorithm=DeflatedHeteroPCA()
                )

                # Compute metrics
                metrics = compute_factor_recovery_metrics(factors, loadings_matrix, true_factors, true_loadings,
                    updated_residuals, true_residuals)
                push!(sim_results, metrics.subspace_distance)
            end

            avg_subspace_dist = mean(sim_results)
            push!(results, avg_subspace_dist)

            # Test that algorithm handles missing data and still achieves reasonable recovery
            @test avg_subspace_dist < 0.5  # Allow degraded performance with missing data
        end

        println("Missing data robustness - subspace distances: $(round.(results, digits=4))")
        println("Missing percentages: $(missing_probs)")
    end
end