using Test, OptimalGIV, DataFrames, CategoricalArrays, StatsModels
using OptimalGIV: create_observation_index, calculate_entity_variance

@testset "ObservationIndex ordering matches StatsModels" begin
    @testset "Numeric IDs" begin
        # Test with non-sequential numeric IDs
        df = DataFrame(
            t = [1, 1, 1, 2, 2, 2],
            id = [10, 5, 3, 10, 5, 3],  # Not in natural order
            y = randn(6),
            x = ones(6),
            weight = ones(6)
        )
        
        # Create observation index
        obs_index = create_observation_index(df, :id, :t)
        
        # Check that internal mapping uses natural sort
        # After sorting by (t, id), the data will be:
        # t=1: [3, 5, 10], t=2: [3, 5, 10]
        # These map to internal IDs: 3→1, 5→2, 10→3
        expected_ids = [1, 2, 3, 1, 2, 3]
        @test obs_index.ids == expected_ids
        
        # Verify internal ID mapping
        @test obs_index.N == 3
        @test obs_index.T == 2
        
        # Entity 3 should map to internal ID 1
        # Entity 5 should map to internal ID 2
        # Entity 10 should map to internal ID 3
        @test obs_index.entity_obs_indices[1, 1] == 1  # Entity 3, time 1
        @test obs_index.entity_obs_indices[2, 1] == 2  # Entity 5, time 1
        @test obs_index.entity_obs_indices[3, 1] == 3  # Entity 10, time 1
    end
    
    @testset "String IDs" begin
        df = DataFrame(
            t = [1, 1, 1, 2, 2, 2],
            id = ["firm_C", "firm_A", "firm_B", "firm_C", "firm_A", "firm_B"],
            y = randn(6),
            x = ones(6),
            weight = ones(6)
        )
        
        # Create observation index
        obs_index = create_observation_index(df, :id, :t)
        
        # Check alphabetical ordering
        # After sorting by (t, id), the data will be:
        # t=1: ["firm_A", "firm_B", "firm_C"], t=2: ["firm_A", "firm_B", "firm_C"]
        expected_ids = [1, 2, 3, 1, 2, 3]  # firm_A→1, firm_B→2, firm_C→3
        @test obs_index.ids == expected_ids
    end
    
    @testset "CategoricalArray IDs" begin
        # Test with categorical array to match actual usage
        df = DataFrame(
            t = [1, 1, 1, 2, 2, 2],
            id = categorical([10, 5, 3, 10, 5, 3]),
            y = randn(6),
            x = ones(6),
            weight = ones(6)
        )
        
        # CategoricalArrays auto-sort levels: [3, 5, 10]
        @test levels(df.id) == [3, 5, 10]
        
        # Create observation index
        obs_index = create_observation_index(df, :id, :t)
        
        # Should match natural sort order
        # Internal IDs: 3→1, 5→2, 10→3
        expected_ids = [1, 2, 3, 1, 2, 3]
        @test obs_index.ids == expected_ids
    end
    
    @testset "Variance array alignment" begin
        # Create data where we can verify variance calculations
        df = DataFrame(
            t = [1, 1, 1, 2, 2, 2],
            id = categorical([10, 5, 3, 10, 5, 3]),
            weight = ones(6)
        )
        
        obs_index = create_observation_index(df, :id, :t)
        
        # Create zero-mean residuals with different variances per entity
        # After sorting by (t,id), order is: [3,5,10,3,5,10]
        # Entity 3 (internal ID 1) has high variance
        # Entity 5 (internal ID 2) has medium variance
        # Entity 10 (internal ID 3) has low variance
        residuals = [-0.3, -0.1, -0.05, 0.3, 0.1, 0.05]  # Zero mean by entity
        
        # Calculate variance by entity
        σu²vec = calculate_entity_variance(residuals, obs_index)
        
        # Verify variances are in the correct order
        @test length(σu²vec) == 3
        # σu²vec[1] should be variance for entity 3 (high variance)
        # σu²vec[2] should be variance for entity 5 (medium variance)
        # σu²vec[3] should be variance for entity 10 (low variance)
        @test σu²vec[1] ≈ ((-0.3)^2 + 0.3^2) / 1  # Entity 3: sum of squares / (n-1)
        @test σu²vec[2] ≈ ((-0.1)^2 + 0.1^2) / 1  # Entity 5
        @test σu²vec[3] ≈ ((-0.05)^2 + 0.05^2) / 1  # Entity 10
    end
    
    @testset "Consistency with StatsModels formula" begin
        # Test that our ordering matches what StatsModels expects
        df = DataFrame(
            t = repeat([1, 2], inner=3),
            id = categorical(repeat([10, 5, 3], 2)),
            y = randn(6),
            x = ones(6),
            weight = ones(6)
        )
        
        # Check StatsModels ordering
        f = @formula(y ~ 0 + id)
        sch = apply_schema(f, StatsModels.schema(f, df))
        coef_names = coefnames(sch)
        
        # Should have coefficients for id: 5 and id: 10 (3 is reference)
        @test "id: 5" in coef_names[2]
        @test "id: 10" in coef_names[2]
        
        # Our observation index should use same ordering
        obs_index = create_observation_index(df, :id, :t)
        # After sorting by (t,id), order is: [3,5,10,3,5,10]
        # Internal IDs: 3→1, 5→2, 10→3
        @test obs_index.ids[1] == 1   # First obs after sort is id=3, maps to internal 1
        @test obs_index.ids[2] == 2   # Second obs after sort is id=5, maps to internal 2
        @test obs_index.ids[3] == 3   # Third obs after sort is id=10, maps to internal 3
    end
end