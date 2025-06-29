using Test, OptimalGIV, DataFrames, Random, Statistics
using HeteroPCA: HeteroPCAModel, predict, reconstruct
using OptimalGIV: vector_to_matrix, matrix_to_vector, extract_pcs_from_residuals, create_observation_index

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