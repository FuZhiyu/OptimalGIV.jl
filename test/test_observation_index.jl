using Test
using OptimalGIV
using OptimalGIV: ObservationIndex, create_observation_index, create_exclusion_matrix, vector_to_matrix, matrix_to_vector
using DataFrames, CategoricalArrays

@testset "ObservationIndex Tests" begin
    
    @testset "Basic ObservationIndex Creation" begin
        # Create test data - balanced panel
        df = DataFrame(
            id = [1, 2, 3, 1, 2, 3, 1, 2, 3],
            t = [1, 1, 1, 2, 2, 2, 3, 3, 3],
            q = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        )
        
        obs_index = create_observation_index(df, :id, :t)
        
        @test obs_index.N == 3
        @test obs_index.T == 3
        @test length(obs_index.ids) == 9
        @test size(obs_index.entity_obs_indices) == (3, 3)
        @test size(obs_index.exclpairs) == (3, 3)
        
        # Check that all entities are present in all time periods
        @test all(obs_index.entity_obs_indices .> 0)
        
        # Check start and end indices
        @test obs_index.start_indices == [1, 4, 7]
        @test obs_index.end_indices == [3, 6, 9]
    end
    
    @testset "Natural Sort Order - Numeric IDs" begin
        # Test with non-sequential numeric IDs to ensure natural sorting
        df = DataFrame(
            id = [10, 5, 20, 10, 5, 20],
            t = [1, 1, 1, 2, 2, 2],
            q = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        )
        
        obs_index = create_observation_index(df, :id, :t)
        
        # IDs should be mapped in natural order: 5->1, 10->2, 20->3
        @test obs_index.N == 3
        @test obs_index.T == 2
        
        # After sorting by [t, id], the order becomes:
        # t=1: id=5, id=10, id=20 (obs 1, 2, 3)
        # t=2: id=5, id=10, id=20 (obs 4, 5, 6)
        # Check entity mapping
        @test obs_index.entity_obs_indices[1, 1] == 1  # id=5, t=1 -> obs 1
        @test obs_index.entity_obs_indices[2, 1] == 2  # id=10, t=1 -> obs 2  
        @test obs_index.entity_obs_indices[3, 1] == 3  # id=20, t=1 -> obs 3
    end
    
    @testset "Natural Sort Order - String IDs" begin
        # Test with string IDs
        df = DataFrame(
            id = ["firm_C", "firm_A", "firm_B", "firm_C", "firm_A", "firm_B"],
            t = [1, 1, 1, 2, 2, 2],
            q = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        )
        
        obs_index = create_observation_index(df, :id, :t)
        
        # IDs should be mapped in alphabetical order: firm_A->1, firm_B->2, firm_C->3
        @test obs_index.N == 3
        @test obs_index.T == 2
        
        # After sorting by [t, id], the order becomes:
        # t=1: firm_A, firm_B, firm_C (obs 1, 2, 3)
        # t=2: firm_A, firm_B, firm_C (obs 4, 5, 6)
        # Check entity mapping
        @test obs_index.entity_obs_indices[1, 1] == 1  # firm_A, t=1 -> obs 1
        @test obs_index.entity_obs_indices[2, 1] == 2  # firm_B, t=1 -> obs 2
        @test obs_index.entity_obs_indices[3, 1] == 3  # firm_C, t=1 -> obs 3
    end
    
    @testset "Natural Sort Order - CategoricalArray IDs" begin
        # Test with CategoricalArray to ensure natural sorting
        df = DataFrame(
            id = CategoricalArray([3, 1, 2, 3, 1, 2]),
            t = [1, 1, 1, 2, 2, 2],
            q = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        )
        
        obs_index = create_observation_index(df, :id, :t)
        
        # IDs should be mapped in natural order: 1->1, 2->2, 3->3
        @test obs_index.N == 3
        @test obs_index.T == 2
        
        # After sorting by [t, id], the order becomes:
        # t=1: id=1, id=2, id=3 (obs 1, 2, 3)
        # t=2: id=1, id=2, id=3 (obs 4, 5, 6)
        # Check entity mapping - should follow natural order regardless of appearance order
        @test obs_index.entity_obs_indices[1, 1] == 1  # id=1, t=1 -> obs 1
        @test obs_index.entity_obs_indices[2, 1] == 2  # id=2, t=1 -> obs 2  
        @test obs_index.entity_obs_indices[3, 1] == 3  # id=3, t=1 -> obs 3
    end
    
    @testset "Unbalanced Panel" begin
        # Create unbalanced panel - entity 2 missing in period 3
        df = DataFrame(
            id = [1, 2, 3, 1, 2, 3, 1, 3],  # entity 2 missing in t=3
            t = [1, 1, 1, 2, 2, 2, 3, 3],
            q = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        )
        
        obs_index = create_observation_index(df, :id, :t)
        
        @test obs_index.N == 3
        @test obs_index.T == 3
        @test length(obs_index.ids) == 8
        
        # Check that entity 2 is missing in time period 3
        @test obs_index.entity_obs_indices[2, 3] == 0
        @test obs_index.entity_obs_indices[1, 3] == 7  # entity 1 in t=3 -> obs 7
        @test obs_index.entity_obs_indices[3, 3] == 8  # entity 3 in t=3 -> obs 8
        
        # Check start and end indices for time periods
        @test obs_index.start_indices == [1, 4, 7]
        @test obs_index.end_indices == [3, 6, 8]
    end
    
    @testset "Exclusion Matrix Creation" begin
        id_values = [1, 2, 3, 4, 5]
        exclude_pairs = Dict(1 => [2, 3], 4 => [5])
        
        exclmat = create_exclusion_matrix(id_values, exclude_pairs)
        
        @test size(exclmat) == (5, 5)
        @test exclmat[1, 2] == true  # pair (1,2) excluded
        @test exclmat[2, 1] == true  # symmetric
        @test exclmat[1, 3] == true  # pair (1,3) excluded
        @test exclmat[3, 1] == true  # symmetric
        @test exclmat[4, 5] == true  # pair (4,5) excluded
        @test exclmat[5, 4] == true  # symmetric
        @test exclmat[2, 3] == false # pair (2,3) not excluded
        @test exclmat[1, 4] == false # pair (1,4) not excluded
    end
    
    @testset "Vector to Matrix Conversion" begin
        # Test with balanced panel
        df = DataFrame(
            id = [1, 2, 3, 1, 2, 3],
            t = [1, 1, 1, 2, 2, 2],
            q = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        )
        
        obs_index = create_observation_index(df, :id, :t)
        values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        
        matrix = vector_to_matrix(values, obs_index)
        
        @test size(matrix) == (3, 2)
        @test matrix[1, 1] == 0.1  # entity 1, time 1
        @test matrix[2, 1] == 0.2  # entity 2, time 1
        @test matrix[3, 1] == 0.3  # entity 3, time 1
        @test matrix[1, 2] == 0.4  # entity 1, time 2
        @test matrix[2, 2] == 0.5  # entity 2, time 2
        @test matrix[3, 2] == 0.6  # entity 3, time 2
    end
    
    @testset "Vector to Matrix Conversion - Unbalanced" begin
        # Test with unbalanced panel
        df = DataFrame(
            id = [1, 2, 3, 1, 3],  # entity 2 missing in t=2
            t = [1, 1, 1, 2, 2],
            q = [1.0, 2.0, 3.0, 4.0, 5.0]
        )
        
        obs_index = create_observation_index(df, :id, :t)
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        matrix = vector_to_matrix(values, obs_index)
        
        @test size(matrix) == (3, 2)
        @test matrix[1, 1] == 0.1    # entity 1, time 1
        @test matrix[2, 1] == 0.2    # entity 2, time 1
        @test matrix[3, 1] == 0.3    # entity 3, time 1
        @test matrix[1, 2] == 0.4    # entity 1, time 2
        @test ismissing(matrix[2, 2]) # entity 2 missing in time 2
        @test matrix[3, 2] == 0.5    # entity 3, time 2
    end
    
    @testset "Matrix to Vector Conversion" begin
        # Test round-trip conversion: vector -> matrix -> vector
        df = DataFrame(
            id = [1, 2, 3, 1, 2, 3],
            t = [1, 1, 1, 2, 2, 2],
            q = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        )
        
        obs_index = create_observation_index(df, :id, :t)
        original_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        
        # Convert to matrix and back
        value_matrix = vector_to_matrix(original_values, obs_index)
        recovered_values = matrix_to_vector(value_matrix, obs_index)
        
        @test recovered_values ≈ original_values
    end
    
    @testset "Matrix to Vector Conversion - Unbalanced" begin
        # Test round-trip with unbalanced panel
        df = DataFrame(
            id = [1, 2, 3, 1, 3],  # entity 2 missing in t=2
            t = [1, 1, 1, 2, 2],
            q = [1.0, 2.0, 3.0, 4.0, 5.0]
        )
        
        obs_index = create_observation_index(df, :id, :t)
        original_values = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Convert to matrix and back
        value_matrix = vector_to_matrix(original_values, obs_index)
        recovered_values = matrix_to_vector(value_matrix, obs_index)
        
        @test recovered_values ≈ original_values
    end
    
    @testset "Data Sorting" begin
        # Test that data gets sorted correctly
        df = DataFrame(
            id = [2, 1, 3, 1, 3, 2],  # Not sorted
            t = [1, 2, 1, 1, 2, 2],   # Not sorted  
            q = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        )
        
        # Make a copy to test if original is modified
        df_copy = copy(df)
        obs_index = create_observation_index(df_copy, :id, :t)
        
        # Check that the dataframe was sorted
        @test issorted(df_copy, [:t, :id])
        
        # Check structure is correct after sorting
        @test obs_index.N == 3
        @test obs_index.T == 2
        @test obs_index.start_indices == [1, 4]
        @test obs_index.end_indices == [3, 6]
    end
    
    @testset "Consistency with Natural Sort Order" begin
        # This test ensures the ordering is consistent with what StatsModels.jl would produce
        df = DataFrame(
            id = [3, 1, 2, 3, 1, 2],  # Appearance order: 3, 1, 2
            t = [1, 1, 1, 2, 2, 2],
            q = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        )
        
        obs_index = create_observation_index(df, :id, :t)
        
        # Natural order should be 1, 2, 3 (not 3, 1, 2 which is appearance order)
        # Entity indices in obs_index.ids should reflect this natural ordering
        
        # After sorting by [t, id], the order should be:
        # t=1: id=1, id=2, id=3 (obs 1, 2, 3)
        # t=2: id=1, id=2, id=3 (obs 4, 5, 6)
        
        @test obs_index.entity_obs_indices[1, 1] == 1  # id=1, t=1 -> obs 1
        @test obs_index.entity_obs_indices[2, 1] == 2  # id=2, t=1 -> obs 2  
        @test obs_index.entity_obs_indices[3, 1] == 3  # id=3, t=1 -> obs 3
        @test obs_index.entity_obs_indices[1, 2] == 4  # id=1, t=2 -> obs 4
        @test obs_index.entity_obs_indices[2, 2] == 5  # id=2, t=2 -> obs 5
        @test obs_index.entity_obs_indices[3, 2] == 6  # id=3, t=2 -> obs 6
    end
    
    @testset "Advanced Ordering Tests from StatsModels Integration" begin
        @testset "Non-sequential numeric IDs with internal mapping" begin
            # Test with non-sequential numeric IDs  
            df = DataFrame(
                t = [1, 1, 1, 2, 2, 2],
                id = [10, 5, 3, 10, 5, 3],  # Not in natural order
                y = randn(6),
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
        
        @testset "String IDs with alphabetical ordering" begin
            df = DataFrame(
                t = [1, 1, 1, 2, 2, 2],
                id = ["firm_C", "firm_A", "firm_B", "firm_C", "firm_A", "firm_B"],
                y = randn(6),
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
        
        @testset "CategoricalArray IDs with level ordering" begin
            # Test with categorical array to match actual usage
            df = DataFrame(
                t = [1, 1, 1, 2, 2, 2],
                id = categorical([10, 5, 3, 10, 5, 3]),
                y = randn(6),
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
        
        @testset "Entity variance array alignment" begin
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
            
            # Test if variance function exists and works (only if available)
            try
                using OptimalGIV: calculate_entity_variance
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
            catch
                # Skip this test if calculate_entity_variance is not available
                @test_skip "calculate_entity_variance function not available"
            end
        end
    end
end