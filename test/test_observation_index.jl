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
    
    @testset "Natural Sort Order" begin
        # Test numeric IDs with natural sorting
        df_numeric = DataFrame(
            id = [10, 5, 20, 10, 5, 20],
            t = [1, 1, 1, 2, 2, 2],
            q = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        )
        
        obs_index = create_observation_index(df_numeric, :id, :t)
        
        # IDs should be mapped in natural order: 5->1, 10->2, 20->3
        @test obs_index.N == 3
        @test obs_index.T == 2
        @test obs_index.entity_obs_indices[1, 1] == 1  # id=5, t=1 -> obs 1
        @test obs_index.entity_obs_indices[2, 1] == 2  # id=10, t=1 -> obs 2  
        @test obs_index.entity_obs_indices[3, 1] == 3  # id=20, t=1 -> obs 3
        
        # Test string IDs with alphabetical ordering
        df_string = DataFrame(
            id = ["firm_C", "firm_A", "firm_B", "firm_C", "firm_A", "firm_B"],
            t = [1, 1, 1, 2, 2, 2],
            q = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        )
        
        obs_index_str = create_observation_index(df_string, :id, :t)
        
        # IDs should be mapped in alphabetical order: firm_A->1, firm_B->2, firm_C->3
        @test obs_index_str.N == 3
        @test obs_index_str.T == 2
        @test obs_index_str.entity_obs_indices[1, 1] == 1  # firm_A, t=1 -> obs 1
        @test obs_index_str.entity_obs_indices[2, 1] == 2  # firm_B, t=1 -> obs 2
        @test obs_index_str.entity_obs_indices[3, 1] == 3  # firm_C, t=1 -> obs 3
        
        # Test categorical arrays with natural ordering
        df_cat = DataFrame(
            id = CategoricalArray([3, 1, 2, 3, 1, 2]),
            t = [1, 1, 1, 2, 2, 2],
            q = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        )
        
        obs_index_cat = create_observation_index(df_cat, :id, :t)
        
        # IDs should be mapped in natural order: 1->1, 2->2, 3->3
        @test obs_index_cat.N == 3
        @test obs_index_cat.T == 2
        @test obs_index_cat.entity_obs_indices[1, 1] == 1  # id=1, t=1 -> obs 1
        @test obs_index_cat.entity_obs_indices[2, 1] == 2  # id=2, t=1 -> obs 2  
        @test obs_index_cat.entity_obs_indices[3, 1] == 3  # id=3, t=1 -> obs 3
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
        df_balanced = DataFrame(
            id = [1, 2, 3, 1, 2, 3],
            t = [1, 1, 1, 2, 2, 2],
            q = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        )
        
        obs_index_balanced = create_observation_index(df_balanced, :id, :t)
        values_balanced = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        
        matrix_balanced = vector_to_matrix(values_balanced, obs_index_balanced)
        
        @test size(matrix_balanced) == (3, 2)
        @test matrix_balanced[1, 1] == 0.1  # entity 1, time 1
        @test matrix_balanced[2, 1] == 0.2  # entity 2, time 1
        @test matrix_balanced[3, 1] == 0.3  # entity 3, time 1
        @test matrix_balanced[1, 2] == 0.4  # entity 1, time 2
        @test matrix_balanced[2, 2] == 0.5  # entity 2, time 2
        @test matrix_balanced[3, 2] == 0.6  # entity 3, time 2
        
        # Test with unbalanced panel
        df_unbalanced = DataFrame(
            id = [1, 2, 3, 1, 3],  # entity 2 missing in t=2
            t = [1, 1, 1, 2, 2],
            q = [1.0, 2.0, 3.0, 4.0, 5.0]
        )
        
        obs_index_unbalanced = create_observation_index(df_unbalanced, :id, :t)
        values_unbalanced = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        matrix_unbalanced = vector_to_matrix(values_unbalanced, obs_index_unbalanced)
        
        @test size(matrix_unbalanced) == (3, 2)
        @test matrix_unbalanced[1, 1] == 0.1    # entity 1, time 1
        @test matrix_unbalanced[2, 1] == 0.2    # entity 2, time 1
        @test matrix_unbalanced[3, 1] == 0.3    # entity 3, time 1
        @test matrix_unbalanced[1, 2] == 0.4    # entity 1, time 2
        @test ismissing(matrix_unbalanced[2, 2]) # entity 2 missing in time 2
        @test matrix_unbalanced[3, 2] == 0.5    # entity 3, time 2
    end
    
    @testset "Matrix to Vector Conversion" begin
        # Test round-trip conversion: vector -> matrix -> vector (balanced)
        df_balanced = DataFrame(
            id = [1, 2, 3, 1, 2, 3],
            t = [1, 1, 1, 2, 2, 2],
            q = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        )
        
        obs_index_balanced = create_observation_index(df_balanced, :id, :t)
        original_values_balanced = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        
        # Convert to matrix and back
        value_matrix_balanced = vector_to_matrix(original_values_balanced, obs_index_balanced)
        recovered_values_balanced = matrix_to_vector(value_matrix_balanced, obs_index_balanced)
        
        @test recovered_values_balanced ≈ original_values_balanced
        
        # Test round-trip with unbalanced panel
        df_unbalanced = DataFrame(
            id = [1, 2, 3, 1, 3],  # entity 2 missing in t=2
            t = [1, 1, 1, 2, 2],
            q = [1.0, 2.0, 3.0, 4.0, 5.0]
        )
        
        obs_index_unbalanced = create_observation_index(df_unbalanced, :id, :t)
        original_values_unbalanced = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Convert to matrix and back
        value_matrix_unbalanced = vector_to_matrix(original_values_unbalanced, obs_index_unbalanced)
        recovered_values_unbalanced = matrix_to_vector(value_matrix_unbalanced, obs_index_unbalanced)
        
        @test recovered_values_unbalanced ≈ original_values_unbalanced
    end
    
    @testset "Data Sorting and Consistency" begin
        # Test that data gets sorted correctly and maintains natural order
        df = DataFrame(
            id = [3, 1, 2, 3, 1, 2],  # Appearance order: 3, 1, 2 (not natural)
            t = [1, 2, 1, 2, 1, 2],   # Not sorted by time first
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
        
        # Natural order should be 1, 2, 3 (not 3, 1, 2 which is appearance order)
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
end