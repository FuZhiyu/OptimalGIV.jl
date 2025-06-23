using Test, OptimalGIV, DataFrames, CategoricalArrays, LinearAlgebra, StatsModels, FixedEffectModels, StatsBase
using OptimalGIV: ols_with_fixed_effects, parse_fe

@testset "ols_with_fixed_effects functionality" begin
    # Simulate data
    N = 50  # number of entities
    T = 20  # number of time periods

    # Create panel structure
    id = repeat(1:N, T)
    t = repeat(1:T, inner=N)

    # Create dataframe
    df = DataFrame(
        id=id,
        t=t
    )

    # Generate random data for covariates
    df.η1 = randn(N * T)
    df.η2 = randn(N * T)

    # Generate true coefficients
    true_β_η1 = randn(N)  # coefficients for id & η1
    true_β_η2 = randn(1)[1]  # coefficient for η2 (with FE)

    # Generate id and time fixed effects
    id_fe = randn(N)
    t_fe = randn(T)

    # Generate y2 variable (to be interacted with id)
    df.y2 = randn(N * T)

    # Generate dependent variable y1 based on the model
    # y1 = id_fe + t_fe + (id & η1) * β_η1 + η2 * β_η2 + error
    df.y1 = zeros(N * T)
    for i in 1:N*T
        id_idx = Int(df.id[i])
        t_idx = Int(df.t[i])
        df.y1[i] = id_fe[id_idx] + t_fe[t_idx] +
                   df.η1[i] * true_β_η1[id_idx] +
                   df.η2[i] * true_β_η2 +
                   0.1 * randn()  # small error term
    end

    # Define formula
    df.id = categorical(df.id)
    f = @formula(y1 + id & y2 ~ id & η1 + fe(id) & η2 + fe(id) + fe(t))

    # Run ols_step
    Y, X, residuals, coef, formula_schema, feM, feids, fekeys, oldY, oldX, coefnames_Y, coefnames_X = ols_with_fixed_effects(df, f)

    # Test 1: Check dimensions
    @test size(Y, 1) == N * T
    @test size(Y, 2) == N + 1  # y1 and id&y2 (N levels)
    @test size(X, 2) == N  # id&η1 (N levels) and η2
    X
    # Test 2: Check that Y contains correct columns
    # First column should be y1 after partialing out fixed effects
    # Other columns should be id&y2 after partialing out fixed effects

    # Test 3: Check that X contains correct columns  
    # Should have id&η1 interactions and η2 after partialing out fixed effects

    # Test 4: Manual OLS calculation
    # After partialing out fixed effects, compute β = (X'X)^(-1)X'Y
    XX = X' * X
    XY = X' * Y
    manual_coef = inv(XX) * XY

    # Compare coefficients
    @test coef ≈ manual_coef atol = 1e-10

    # Test 5: Check residuals
    manual_residuals = Y - X * coef
    @test residuals ≈ manual_residuals atol = 1e-10

    # Test 6: Verify residuals are orthogonal to X
    @test maximum(abs.(X' * residuals)) < 1e-10

    # Test 7: Check that fixed effects were properly removed
    # The mean of each variable within id and t groups should be close to zero
    gdf_id = groupby(DataFrame(Y1=Y[:, 1], id=df.id), :id)
    id_means = combine(gdf_id, :Y1 => mean => :mean_Y1)
    @test maximum(abs.(id_means.mean_Y1)) < 1e-5

    gdf_t = groupby(DataFrame(Y1=Y[:, 1], t=df.t), :t)
    t_means = combine(gdf_t, :Y1 => mean => :mean_Y1)
    @test maximum(abs.(t_means.mean_Y1)) < 1e-5
end

