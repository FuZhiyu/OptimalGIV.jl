using Test, OptimalGIV, DataFrames, CategoricalArrays, StatsModels, Random
using OptimalGIV: preprocess_dataframe, get_coefnames, parse_guess, PCTerm, has_pc, get_pc_k, remove_pc_terms, separate_giv_ols_fe_formulas, create_coef_dataframe
using StatsModels: term, ConstantTerm, InterceptTerm, CategoricalTerm, InteractionTerm
df = DataFrame(
    id=categorical(repeat(1:10, outer=50)),
    t=repeat(1:50, inner=10),
    S=rand(10 * 50),
    q=rand(10 * 50),
    p=repeat(rand(50), inner=10),
    ζ=rand(10 * 50),
    u=rand(10 * 50),
    η=rand(10 * 50)
)
df = DataFrames.shuffle(df)


##============= test preprocess_dataframe =============##
df2 = preprocess_dataframe(df, @formula(q + id & endog(p) ~ η), :id, :t, :S)
@test issorted(df2, [:t, :id])

##============= test error for IDs with less than 2 observations =============##
# Create a DataFrame where some IDs have less than 2 observations
df_insufficient = DataFrame(
    id=categorical([1, 1, 2, 3, 3, 3]),  # ID 2 has only 1 observation
    t=[1, 2, 1, 1, 2, 3],
    S=[0.5, 0.5, 1.0, 0.33, 0.33, 0.34],
    q=rand(6),
    p=rand(6),
    η=rand(6)
)
# Test that it throws an ArgumentError
@test_throws ArgumentError preprocess_dataframe(df_insufficient, @formula(q + id & endog(p) ~ η), :id, :t, :S)

##============= test get_coefnames =============##
@testset "test get_coefnames" begin
    # Test with DataFrame and formula (public interface)
    df_coef = DataFrame(
        id=categorical(repeat(1:3, outer=3)),
        t=repeat(1:3, inner=3),
        S=repeat([0.33, 0.33, 0.34], 3),
        q=rand(9),
        p=rand(9),
        η=rand(9),
        C1=repeat([1, 2, 3], 3),
        C2=categorical(repeat(["A", "B", "C"], 3))
    )

    # Test 1: Basic formula with categorical interaction
    formula1 = @formula(q + id & endog(p) ~ η)
    response_name, endog_name, endog_coefnames, covariates_names, slope_terms = get_coefnames(df_coef, formula1)
    @test response_name == "q"
    @test endog_name == "p"
    @test endog_coefnames == ["id: 1 & p", "id: 2 & p", "id: 3 & p"]
    @test covariates_names == ["(Intercept)", "η"]
    # Test 2: Formula with continuous variable interaction
    formula2 = @formula(q + C1 & endog(p) ~ η)
    response_name, endog_name, endog_coefnames, covariates_names, slope_terms = get_coefnames(df_coef, formula2)
    @test response_name == "q"
    @test endog_name == "p"
    @test endog_coefnames == ["C1 & p"]
    @test covariates_names == ["(Intercept)", "η"]

    # Test 3: Formula with multiple interactions
    formula3 = @formula(q + id & endog(p) + C1 & endog(p) ~ η + id)
    response_name, endog_name, endog_coefnames, covariates_names, slope_terms = get_coefnames(df_coef, formula3)
    @test response_name == "q"
    @test endog_name == "p"
    @test endog_coefnames == ["id: 1 & p", "id: 2 & p", "id: 3 & p", "C1 & p"]
    @test covariates_names == ["(Intercept)", "η", "id: 2", "id: 3"]

    # Test 4: Formula with intercept
    formula4 = @formula(q + endog(p) ~ 0 + id)
    response_name, endog_name, endog_coefnames, covariates_names, slope_terms = get_coefnames(df_coef, formula4)
    @test response_name == "q"
    @test endog_name == "p"
    @test endog_coefnames == ["p"]
    @test covariates_names == ["id: 1", "id: 2", "id: 3"]

    # Test 5: Formula with categorical variable interaction with endogenous
    formula5 = @formula(q + C2 & endog(p) ~ η + 0)
    response_name, endog_name, endog_coefnames, covariates_names, slope_terms = get_coefnames(df_coef, formula5)
    @test response_name == "q"
    @test endog_name == "p"
    @test endog_coefnames == ["C2: A & p", "C2: B & p", "C2: C & p"]
    @test covariates_names == ["η"]

    # Test 6: Formula with no interactions (constant elasticity)
    formula6 = @formula(q + endog(p) ~ η + 0)
    response_name, endog_name, endog_coefnames, covariates_names, slope_terms = get_coefnames(df_coef, formula6)
    @test response_name == "q"
    @test endog_name == "p"
    @test endog_coefnames == ["p"]
    @test covariates_names == ["η"]

    # Test 7: Formula with fixed effects
    formula7 = @formula(q + endog(p) ~ η + fe(id))
    response_name, endog_name, endog_coefnames, covariates_names, slope_terms = get_coefnames(df_coef, formula7)
    @test covariates_names == ["η"]
end

##============= test parse_guess =============##
@testset "test parse_guess" begin
    # Mock elasticity names for testing
    endog_coefnames_single = ["p"]
    endog_coefnames_multi = ["id: 1 & p", "id: 2 & p", "C1 & p"]

    # Test 1: Vector of numbers - should return as is
    guess_vec = [1.0, 2.0, 3.0]
    @test parse_guess(endog_coefnames_multi, guess_vec, Val(:iv)) == guess_vec
    @test parse_guess(endog_coefnames_multi, guess_vec, Val(:debiased_ols)) == guess_vec

    # Test 2: Nothing - should return nothing
    @test isnothing(parse_guess(endog_coefnames_single, nothing, Val(:iv)))
    @test isnothing(parse_guess(endog_coefnames_multi, nothing, Val(:debiased_ols)))

    # Test 3: Single number - should wrap in vector
    @test parse_guess(endog_coefnames_single, 2.5, Val(:iv)) == [2.5]
    @test parse_guess(endog_coefnames_single, -1.2, Val(:debiased_ols)) == [-1.2]

    # Test 4: Dict for scalar_search algorithm - needs "Aggregate" key
    guess_dict_scalar = Dict("Aggregate" => 0.8)
    @test parse_guess(endog_coefnames_single, guess_dict_scalar, Val(:scalar_search)) == [0.8]

    guess_dict_scalar_vec = Dict("Aggregate" => [0.8])
    @test parse_guess(endog_coefnames_single, guess_dict_scalar_vec, Val(:scalar_search)) == [0.8]

    # Test 5: Dict for scalar_search without "Aggregate" key - should throw error
    guess_dict_wrong = Dict("id: 1 & p" => 1.0)
    @test_throws ArgumentError parse_guess(endog_coefnames_single, guess_dict_wrong, Val(:scalar_search))

    # Test 6: Dict for other algorithms - maps elasticity names to values
    guess_dict = Dict(
        "id: 1 & p" => 1.0,
        "id: 2 & p" => 2.0,
        "C1 & p" => 0.5
    )
    result = parse_guess(endog_coefnames_multi, guess_dict, Val(:iv))
    @test result == [1.0, 2.0, 0.5]

    # Test 7: Dict with missing elasticity name - should throw error
    guess_dict_incomplete = Dict(
        "id: 1 & p" => 1.0,
        "id: 2 & p" => 2.0
        # Missing "C1 & p"
    )
    @test_throws ArgumentError parse_guess(endog_coefnames_multi, guess_dict_incomplete, Val(:iv))

    # Test 8: Dict with symbol keys should also work
    guess_dict_symbols = Dict(
        Symbol("id: 1 & p") => 1.5,
        Symbol("id: 2 & p") => 2.5,
        Symbol("C1 & p") => 3.5
    )
    result_symbols = parse_guess(endog_coefnames_multi, guess_dict_symbols, Val(:debiased_ols))
    @test result_symbols == [1.5, 2.5, 3.5]

    # Test 9: Dict with extra keys (should still work - only uses needed ones)
    guess_dict_extra = Dict(
        "id: 1 & p" => 1.0,
        "id: 2 & p" => 2.0,
        "C1 & p" => 0.5,
        "extra_key" => 999.0  # This should be ignored
    )
    result_extra = parse_guess(endog_coefnames_multi, guess_dict_extra, Val(:iv))
    @test result_extra == [1.0, 2.0, 0.5]

    # Test 10: Single elasticity with dict
    guess_dict_single = Dict("p" => 0.7)
    @test parse_guess(endog_coefnames_single, guess_dict_single, Val(:iv)) == [0.7]

    # Test 11: Empty elasticity names (edge case)
    endog_coefnames_empty = String[]
    @test parse_guess(endog_coefnames_empty, Float64[], Val(:iv)) == Float64[]
    @test isnothing(parse_guess(endog_coefnames_empty, nothing, Val(:iv)))
    @test parse_guess(endog_coefnames_empty, Dict{String,Float64}(), Val(:iv)) == Float64[]
end

@testset "PC Interface Tests" begin
    @testset "PCTerm construction and properties" begin
        # Test pc() function creates correct PCTerm
        pc3 = pc(3)
        @test pc3 isa PCTerm{3}

        pc5 = pc(5)
        @test pc5 isa PCTerm{5}

        # Test StatsModels interface
        @test StatsModels.terms(pc3) == []
        @test StatsModels.termvars(pc3) == []
    end

    @testset "has_pc detection" begin
        # Test has_pc for different term types
        @test has_pc(pc(3)) == true
        @test has_pc(term(:x)) == false
        @test has_pc(ConstantTerm(1)) == false

        # Test formula with pc terms
        formula_with_pc = @formula(y ~ x + pc(3))
        @test has_pc(formula_with_pc) == true

        formula_without_pc = @formula(y ~ x + z)
        @test has_pc(formula_without_pc) == false

        # Test tuple detection
        rhs_terms = (term(:x), pc(2), term(:z))
        @test has_pc(rhs_terms) == true

        rhs_terms_no_pc = (term(:x), term(:z))
        @test has_pc(rhs_terms_no_pc) == false
    end

    @testset "get_pc_k extraction" begin
        # Test getting k from PCTerm
        @test get_pc_k(pc(3)) == 3
        @test get_pc_k(pc(7)) == 7

        # Test formula with pc terms
        formula_pc3 = @formula(y ~ x + pc(3))
        @test get_pc_k(formula_pc3) == 3

        formula_no_pc = @formula(y ~ x + z)
        @test get_pc_k(formula_no_pc) == 0

        # Test multiple pc terms (should return maximum)
        rhs_terms_multi = (term(:x), pc(2), pc(5), term(:z))
        @test get_pc_k(rhs_terms_multi) == 5
    end

    @testset "remove_pc_terms functionality" begin
        # Test removing PC terms from formula
        formula_with_pc = @formula(y ~ x + pc(3) + z)
        formula_no_pc = remove_pc_terms(formula_with_pc)

        @test !has_pc(formula_no_pc)

        # Check that other terms are preserved
        rhs_termvars = StatsModels.termvars(formula_no_pc.rhs)
        @test :x in rhs_termvars
        @test :z in rhs_termvars

        # Test formula with only PC term - should become intercept-only (y ~ 1)
        formula_only_pc = @formula(y ~ pc(3))
        formula_intercept = remove_pc_terms(formula_only_pc)
        @test formula_intercept.rhs isa InterceptTerm{true}
        @test string(formula_intercept) == "y ~ 1"

        # Test explicit no-intercept with PC should preserve no-intercept
        formula_no_int_pc = @formula(y ~ 0 + pc(3))
        formula_no_int = remove_pc_terms(formula_no_int_pc)
        @test formula_no_int.rhs isa ConstantTerm{Int64}
        @test formula_no_int.rhs.n == 0
        @test string(formula_no_int) == "y ~ 0"
    end

    @testset "Integration with separate_giv_ols_fe_formulas" begin
        # Create test data
        df = DataFrame(
            id=repeat(1:5, outer=10),
            t=repeat(1:10, inner=5),
            S=rand(50),
            q=rand(50),
            p=repeat(rand(10), inner=5),
            x=rand(50)
        )

        # Test formula with PC terms
        formula_with_pc = @formula(q + endog(p) ~ x + pc(3))

        # This should not throw an error and should return n_pcs = 3
        result = separate_giv_ols_fe_formulas(df, formula_with_pc)
        @test length(result) == 6  # Should now return 6 elements including n_pcs

        formula_givcore, formula_schema, fes, feids, fekeys, n_pcs = result
        @test n_pcs == 3

        # Check that PC terms are removed from formula_schema
        @test !has_pc(formula_schema)

        # Test formula without PC terms
        formula_no_pc = @formula(q + endog(p) ~ x)
        result_no_pc = separate_giv_ols_fe_formulas(df, formula_no_pc)
        formula_givcore_no_pc, formula_schema_no_pc, fes_no_pc, feids_no_pc, fekeys_no_pc, n_pcs_no_pc = result_no_pc
        @test n_pcs_no_pc == 0
    end

    @testset "Error handling and edge cases" begin
        # Test invalid k values - should work at construction but might fail later
        @test pc(0) isa PCTerm{0}
        @test pc(1) isa PCTerm{1}

        # Test very large k values
        @test pc(100) isa PCTerm{100}

        # Test formula parsing with complex expressions
        df = DataFrame(
            id=repeat(1:3, outer=5),
            t=repeat(1:5, inner=3),
            S=rand(15),
            q=rand(15),
            p=repeat(rand(5), inner=3),
            x1=rand(15),
            x2=rand(15)
        )

        # Multiple variables with PC
        formula_complex = @formula(q + id & endog(p) ~ x1 + x2 + pc(2))
        result_complex = separate_giv_ols_fe_formulas(df, formula_complex)
        @test length(result_complex) == 6
        @test result_complex[6] == 2  # n_pcs should be 2
    end
end

##============= test PC ordering =============##
@testset "PC Ordering Tests" begin
    
    @testset "PC loadings match entity ordering" begin
        # Create a simple balanced panel with known structure
        Random.seed!(123)
        
        # Create data where entities appear in non-sorted order
        df = DataFrame(
            id = categorical(repeat([3, 1, 4, 2], 5)),  # Non-sorted entity order
            t = repeat(1:5, inner=4),
            q = randn(20),
            p = randn(20),
            S = ones(20)
        )
        
        # Add simple covariates
        df.x = randn(20)
        
        # Test with pc(1) - simpler case
        model = giv(df, @formula(q + id & endog(p) ~ fe(id) + x + pc(1)), 
                    :id, :t, :S; 
                    algorithm = :iv,
                    guess = [1.0, 1.5, 2.0, 2.5],  # 4 entities
                    save_df = true,
                    quiet = true,  # Suppress convergence warnings
                    iterations = 1)  # Only one iteration since we're just testing structure
        
        # Verify the saved dataframe exists
        @test !isnothing(model.df)
        @test "pc_loading_1" in names(model.df)
        @test "pc_factor_1" in names(model.df)
        
        # Check that each entity has consistent loading across time
        for entity_id in unique(df.id)
            entity_rows = model.df[model.df.id .== entity_id, :]
            loadings = entity_rows.pc_loading_1
            # All loadings for the same entity should be identical
            @test all(loadings .≈ loadings[1])
        end
        
        # Check that each time period has consistent factor across entities
        for time_period in unique(df.t)
            time_rows = model.df[model.df.t .== time_period, :]
            factors = time_rows.pc_factor_1
            # All factors for the same time should be identical
            @test all(factors .≈ factors[1])
        end
        
        # Verify we have the correct number of unique loadings and factors
        unique_loadings = unique(round.(model.df.pc_loading_1, digits=10))
        unique_factors = unique(round.(model.df.pc_factor_1, digits=10))
        @test length(unique_loadings) == 4  # 4 entities
        @test length(unique_factors) == 5   # 5 time periods
    end
    
    @testset "PC ordering with very unbalanced panel" begin
        # Create unbalanced panel where different entities appear at different times
        df = DataFrame()
        
        # Entity 5 appears only in periods 1-2
        append!(df, DataFrame(id = categorical(fill(5, 2)), t = [1, 2], 
                            q = randn(2), p = randn(2), S = ones(2), x = randn(2)))
        
        # Entity 1 appears in all periods 1-3
        append!(df, DataFrame(id = categorical(fill(1, 3)), t = [1, 2, 3], 
                            q = randn(3), p = randn(3), S = ones(3), x = randn(3)))
        
        # Entity 3 appears only in period 2-3
        append!(df, DataFrame(id = categorical(fill(3, 2)), t = [2, 3], 
                            q = randn(2), p = randn(2), S = ones(2), x = randn(2)))
        
        # Entity 2 appears in all periods
        append!(df, DataFrame(id = categorical(fill(2, 3)), t = [1, 2, 3], 
                            q = randn(3), p = randn(3), S = ones(3), x = randn(3)))
        
        # Entities appear in order: 5, 1, 3, 2 (not sorted)
        # But ObservationIndex should use sorted order: 1, 2, 3, 5
        
        model = giv(df, @formula(q + id & endog(p) ~ fe(id) + x + pc(1)), 
                    :id, :t, :S; 
                    algorithm = :iv,
                    guess = [1.0, 1.5, 2.0, 2.5],  # 4 entities
                    save_df = true,
                    quiet = true,  # Suppress convergence warnings
                    iterations = 1)  # Only one iteration since we're just testing structure
        
        # Verify loadings are consistent for each entity
        @test !isnothing(model.df)
        for entity_id in unique(df.id)
            entity_rows = model.df[model.df.id .== entity_id, :]
            if nrow(entity_rows) > 0
                loadings = entity_rows.pc_loading_1
                @test all(loadings .≈ loadings[1])
            end
        end
        
        # The key test: verify that the number of unique PC loadings matches number of entities
        unique_loadings = unique(round.(model.df.pc_loading_1, digits=10))
        @test length(unique_loadings) == 4  # Should be 4 distinct loadings for 4 entities
    end
end

##============= test create_coef_dataframe =============##
@testset "create_coef_dataframe comprehensive tests" begin
    
    @testset "Categorical and interaction terms" begin
        # Create test data
        df = DataFrame(
            id = categorical(repeat(1:3, outer=2)),
            t = repeat(1:2, inner=3),
            group = categorical(repeat(["A", "B", "A"], 2)),
            region = categorical(repeat(["X", "Y"], inner=3)),
            q = randn(6),
            p = randn(6),
            x1 = randn(6),
            x2 = randn(6)
        )
        
        # Test 1: No categorical terms (single row output) - tests the crossjoin bug fix
        formula = @formula(q + endog(p) ~ x1 + x2)
        _, formula_schema, _, _, _, _ = separate_giv_ols_fe_formulas(df, formula)
        
        coef = [0.5, 1.0, 2.0, 3.0]  # p_coef, intercept, x1_coef, x2_coef
        coefdf = create_coef_dataframe(df, formula_schema, coef, :id)
        
        @test nrow(coefdf) == 1
        @test ncol(coefdf) == 4  # p_coef, (Intercept)_coef, x1_coef, x2_coef
        @test coefdf.p_coef[1] ≈ 0.5
        @test coefdf.x1_coef[1] ≈ 2.0
        @test coefdf.x2_coef[1] ≈ 3.0
        
        # Test 2: Single categorical term
        formula = @formula(q + id & endog(p) ~ x1)
        _, formula_schema, _, _, _, _ = separate_giv_ols_fe_formulas(df, formula)
        
        coef = [1.0, 1.5, 2.0, 0.5, 3.0]  # 3 id coeffs, intercept, x1
        coefdf = create_coef_dataframe(df, formula_schema, coef, :id)
        
        @test nrow(coefdf) == 3
        @test "id" in names(coefdf)
        @test "id & p_coef" in names(coefdf)
        @test "x1_coef" in names(coefdf)
        @test all(coefdf.x1_coef .≈ 3.0)
        
        # Test 3: Multiple categorical interactions with full dummy coding
        formula = @formula(q + id & group & endog(p) ~ 0 + x1)
        _, formula_schema, _, _, _, _ = separate_giv_ols_fe_formulas(df, formula)
        
        # With full dummy coding (no intercept), id (3) × group (2) = 6 coefficients
        coef = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 3.0]  # 6 interaction coeffs, x1
        coefdf = create_coef_dataframe(df, formula_schema, coef, :id)
        
        # Should have all unique combinations that appear in the data
        unique_combos = unique(select(df, :id, :group))
        @test nrow(coefdf) == nrow(unique_combos)  
        @test "id" in names(coefdf)
        @test "group" in names(coefdf)
        @test "id & group & p_coef" in names(coefdf)
        
        # Test 4: Mixed categorical and continuous interaction
        formula = @formula(q + id & x1 & endog(p) ~ x2)
        _, formula_schema, _, _, _, _ = separate_giv_ols_fe_formulas(df, formula)
        
        coef = [1.0, 1.5, 2.0, 0.5, 3.0]  # 3 id coeffs, intercept, x2
        coefdf = create_coef_dataframe(df, formula_schema, coef, :id)
        
        @test nrow(coefdf) == 3
        @test "id" in names(coefdf)
        @test "id & x1 & p_coef" in names(coefdf)
        @test all(coefdf.x2_coef .≈ 3.0)
    end
    
    @testset "InterceptTerm handling" begin
        df = DataFrame(
            id = categorical([1, 2, 3]),
            q = [1.0, 2.0, 3.0],
            p = [0.5, 1.0, 1.5],
            x = [0.1, 0.2, 0.3]
        )
        
        # Test with no intercept (InterceptTerm{false})
        formula = @formula(q + endog(p) ~ 0 + x)
        _, formula_schema, _, _, _, _ = separate_giv_ols_fe_formulas(df, formula)
        
        coef = [0.5, 2.0]  # p_coef, x_coef (no intercept)
        coefdf = create_coef_dataframe(df, formula_schema, coef, :id)
        
        @test nrow(coefdf) == 1
        @test ncol(coefdf) == 2  # Only p_coef and x_coef
        @test !("(Intercept)_coef" in names(coefdf))
        @test coefdf.p_coef[1] ≈ 0.5
        @test coefdf.x_coef[1] ≈ 2.0
    end
    
    
    
    @testset "Coefficient count validation" begin
        df = DataFrame(
            id = categorical([1, 2, 3]),
            q = [1.0, 2.0, 3.0],
            p = [0.5, 1.0, 1.5],
            x = [0.1, 0.2, 0.3]
        )
        
        formula = @formula(q + id & endog(p) ~ x)
        _, formula_schema, _, _, _, _ = separate_giv_ols_fe_formulas(df, formula)
        
        # Test too many coefficients throws error
        coef_long = ones(10)  # Way more than needed
        @test_throws ArgumentError create_coef_dataframe(df, formula_schema, coef_long, :id)
    end
    
    
    @testset "Fixed effects integration (fekeys)" begin
        df = DataFrame(
            id = categorical(repeat(1:3, outer=4)),
            t = categorical(repeat(1:4, inner=3)),
            group = categorical(repeat(["A", "B", "A"], 4)),
            q = randn(12),
            p = randn(12),
            x = randn(12)
        )
        
        # Simulate formula with fixed effects
        formula = @formula(q + group & endog(p) ~ x + fe(id) + fe(t))
        _, formula_schema, _, _, fekeys, _ = separate_giv_ols_fe_formulas(df, formula)
        
        # fekeys should contain [:id, :t]
        @test :id in fekeys
        @test :t in fekeys
        
        # group has 2 unique values (A, B)
        # With default dummy coding, only B gets a coefficient
        coef = [1.0, 0.5, 2.0]  # 1 group coeff (B), intercept, x
        coefdf = create_coef_dataframe(df, formula_schema, coef, :id; fekeys=fekeys)
        
        # Should include fe keys in categorical_terms_symbol
        @test "id" in names(coefdf)
        @test "t" in names(coefdf)
        @test "group" in names(coefdf)
        
        # Should have unique combinations of all categorical variables including FEs
        expected_rows = nrow(unique(select(df, :id, :t, :group)))
        @test nrow(coefdf) == expected_rows
    end
    
    
    
    @testset "Continuous-only interactions" begin
        df = DataFrame(
            id = categorical([1, 2, 3]),
            q = [1.0, 2.0, 3.0],
            p = [0.5, 1.0, 1.5],
            x1 = [0.1, 0.2, 0.3],
            x2 = [0.2, 0.4, 0.6]
        )
        
        # Interaction of continuous variables only
        formula = @formula(q + x1 & endog(p) ~ x2)
        _, formula_schema, _, _, _, _ = separate_giv_ols_fe_formulas(df, formula)
        
        coef = [0.5, 1.0, 2.0]  # x1&p_coef, intercept, x2_coef
        coefdf = create_coef_dataframe(df, formula_schema, coef, :id)
        
        @test nrow(coefdf) == 1  # No categorical terms
        @test "x1 & p_coef" in names(coefdf)
        @test coefdf[1, "x1 & p_coef"] ≈ 0.5
    end
end

##============= test save logic =============##
@testset "Save Logic Tests" begin
    # Helper function to create test panel data
    function create_test_panel()
        Random.seed!(123)
        DataFrame(
            id = categorical(repeat(1:4, outer=3)),
            t = repeat(1:3, inner=4),
            group = categorical(repeat(["A", "B"], inner=2, outer=3)),
            S = repeat([0.25, 0.25, 0.25, 0.25], 3),
            q = randn(12),
            p = repeat(randn(3), inner=4),
            x1 = randn(12),
            x2 = randn(12)
        )
    end
    
    @testset "Save options" begin
        df_base = create_test_panel()
        formula = @formula(q + id & endog(p) ~ x1 + fe(t))
        
        # Test each save option
        for (save_opt, check_fe, check_res) in [
            (:fe, true, false),
            (:residuals, false, true),
            (:all, true, true),
            (:none, false, false)
        ]
            model = giv(df_base, formula, :id, :t, :S; 
                       algorithm=:iv, guess=[1.0, 1.5, 2.0, 2.5], 
                       save=save_opt, quiet=true)
            
            # Check fixed effects
            if check_fe
                @test !isnothing(model.fe)
                @test "t" in names(model.fe)
                @test nrow(model.fe) == 3
                @test "t" in names(model.coefdf)
            else
                @test isnothing(model.fe)
            end
            
            # Check residuals
            if check_res
                @test !isnothing(model.residual_df)
                @test nrow(model.residual_df) == nrow(df_base)
                @test "q_residual" in names(model.residual_df)
                @test all(isfinite.(model.residual_df.q_residual))
            else
                @test isnothing(model.residual_df)
            end
        end
        
        # Test edge case: no fixed effects but save=:fe
        formula_no_fe = @formula(q + endog(p) ~ x1 + x2)
        model_no_fe = giv(df_base, formula_no_fe, :id, :t, :S; 
                         algorithm=:iv, guess=0.5, 
                         save=:fe, quiet=true)
        @test isnothing(model_no_fe.fe)
    end
    
    @testset "save_df=true tests" begin
        df_base = create_test_panel()
        # Test 1: Basic save_df with categorical elasticities
        formula = @formula(q + id & endog(p) ~ x1 + fe(t))
        model_savedf = giv(df_base, formula, :id, :t, :S; 
                          algorithm=:iv, guess=[1.0, 1.5, 2.0, 2.5], 
                          save_df=true, save=:all, quiet=true)
        
        @test !isnothing(model_savedf.df)
        @test nrow(model_savedf.df) == nrow(df_base)
        @test "q_residual" in names(model_savedf.df)
        @test "id & p_coef" in names(model_savedf.df)
        @test "fe_t" in names(model_savedf.df)  # Fixed effect column
        
        # Test 2: save_df with no categorical terms (crossjoin case)
        formula_no_cat = @formula(q + endog(p) ~ x1 + x2)
        model_no_cat = giv(df_base, formula_no_cat, :id, :t, :S; 
                          algorithm=:iv, guess=0.5, 
                          save_df=true, quiet=true)
        
        @test !isnothing(model_no_cat.df)
        @test nrow(model_no_cat.df) == nrow(df_base)
        @test "p_coef" in names(model_no_cat.df)
        @test all(model_no_cat.df.p_coef .== model_no_cat.df.p_coef[1])  # All same value
        
        # Test 3: save_df with PC terms
        formula_pc = @formula(q + id & endog(p) ~ x1 + pc(2))
        model_pc = giv(df_base, formula_pc, :id, :t, :S; 
                      algorithm=:iv, guess=[1.0, 1.5, 2.0, 2.5], 
                      save_df=true, quiet=true)
        
        @test !isnothing(model_pc.df)
        @test "pc_factor_1" in names(model_pc.df)
        @test "pc_factor_2" in names(model_pc.df)
        @test "pc_loading_1" in names(model_pc.df)
        @test "pc_loading_2" in names(model_pc.df)
    end
    
    @testset "Edge case: FE and coefficient categories don't overlap" begin
        # Create data where group is used for coefficients and id for fixed effects
        df_edge = DataFrame(
            id = categorical(repeat(1:3, outer=4)),
            t = repeat(1:4, inner=3),
            group = categorical(repeat(["X", "Y", "Z"], 4)),
            S = repeat([0.33, 0.33, 0.34], 4),
            q = randn(12),
            p = repeat(randn(4), inner=3),
            x = randn(12)
        )
        
        # Use group for elasticities but id for fixed effects
        formula_edge = @formula(q + group & endog(p) ~ x + fe(id))
        model_edge = giv(df_edge, formula_edge, :id, :t, :S; 
                        algorithm=:iv, guess=[1.0, 1.5, 2.0], 
                        save=:fe, save_df=true, quiet=true)
        
        # Check that both group coefficients and id fixed effects are present
        @test "group" in names(model_edge.coefdf)
        @test "id" in names(model_edge.coefdf)  # Added via fedf join
        @test !isnothing(model_edge.fe)
        @test nrow(model_edge.fe) == 3  # 3 unique ids
        
        # In savedf, should have both
        @test "group" in names(model_edge.df)
        @test "id" in names(model_edge.df)
        @test "group & p_coef" in names(model_edge.df)
        
        # Test another edge case: no overlap at all between formula variables
        # Create data where p varies to avoid collinearity with fixed effects
        df_edge2 = DataFrame(df_edge)
        df_edge2.p = randn(nrow(df_edge2))
        formula_edge2 = @formula(q + endog(p) ~ x + fe(id) + fe(t))
        model_edge2 = giv(df_edge2, formula_edge2, :id, :t, :S; 
                         algorithm=:iv, guess=0.5, 
                         save=:all, save_df=true, quiet=true)
        
        # coefdf should have no categorical columns initially (crossjoin case)
        # but after FE merge, should have id and t
        @test "id" in names(model_edge2.coefdf)
        @test "t" in names(model_edge2.coefdf)
        @test "p_coef" in names(model_edge2.df)
        @test "q_residual" in names(model_edge2.df)
    end
    
end