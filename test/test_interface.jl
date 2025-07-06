using Test, OptimalGIV, DataFrames, CategoricalArrays, StatsModels
using OptimalGIV: preprocess_dataframe, get_coefnames, parse_guess, PCTerm, has_pc, get_pc_k, remove_pc_terms, separate_giv_ols_fe_formulas
using StatsModels: term, ConstantTerm, InterceptTerm
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