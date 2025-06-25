using Test, OptimalGIV, DataFrames, CategoricalArrays
using OptimalGIV: preprocess_dataframe, get_coefnames, parse_guess
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
