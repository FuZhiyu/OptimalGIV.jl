using Test, OptimalGIV, Random, LinearAlgebra, Statistics
using OptimalGIV: calculate_entity_variance, create_observation_index, full_column_rank,
    solve_ols_vcov, solve_optimal_vcov
using DataFrames, CategoricalArrays

include("vcov_test_helpers.jl")   # _load_simdata1

# Panel sorted by (t, id); `present[i]` lists the periods entity i is observed.
function _unbalanced_index(present, T)
    rows = sort!(DataFrame(
            t=reduce(vcat, [collect(p) for p in present]),
            id=reduce(vcat, [fill(i, length(p)) for (i, p) in enumerate(present)])),
        [:t, :id])
    rows.id = CategoricalArray(rows.id)
    return rows, create_observation_index(rows, :id, :t)
end

# Every entity's X'X scaled by 1/nᵢ, then the sandwich divided by common T.
function _common_t_ols_vcov(σu², X, obs_index)
    bread, meat = zeros(size(X, 2), size(X, 2)), zeros(size(X, 2), size(X, 2))
    for i in 1:obs_index.N
        rows = filter(>(0), obs_index.entity_obs_indices[i, :])
        XX = X[rows, :]' * X[rows, :] / length(rows)
        bread .+= XX
        meat .+= XX .* σu²[i]
    end
    return inv(bread) * meat * inv(bread) / obs_index.T
end

@testset "solve_ols_vcov: analytic unbalanced panels" begin
    T = 40
    present = [1:40, 1:30, 11:20, 21:40]
    n = length.(present)
    σu² = [0.5, 1.2, 2.0, 0.8]
    df, obs_index = _unbalanced_index(present, T)
    ids = Int.(levelcode.(df.id))

    # Entity intercept: σᵢ²/nᵢ, not the σᵢ²/T of the common-T scaling.
    entity_dummies = [Float64(ids[r] == i) for r in eachindex(ids), i in 1:4]
    @test solve_ols_vcov(σu², entity_dummies, obs_index) ≈ Diagonal(σu² ./ n) rtol = 1e-12
    @test diag(_common_t_ols_vcov(σu², entity_dummies, obs_index)) ≈ σu² ./ T rtol = 1e-12

    # Regime-split entity intercepts: σᵢ²/nᵢᵣ within each occupied regime.
    post = df.t .> 20
    regime_columns = [entity_dummies .* .!post entity_dummies .* post]
    occupied = vec(any(!iszero, regime_columns; dims=1))
    nᵢᵣ = vec(sum(regime_columns; dims=1))[occupied]
    kᵢ = [count(occupied[[i, i + 4]]) for i in 1:4]   # one or two fitted intercepts
    σ̃² = σu² .* (n .- 1) ./ (n .- kᵢ)
    @test diag(solve_ols_vcov(σu², regime_columns[:, occupied], obs_index)) ≈
          repeat(σ̃², 2)[occupied] ./ nᵢᵣ rtol = 1e-12

    # One pooled intercept: each entity's leverage is its share of the rows.
    pooled = ones(length(ids), 1)
    σ̃²_pooled = σu² .* (n .- 1) ./ (n .- n ./ sum(n))
    @test only(solve_ols_vcov(σu², pooled, obs_index)) ≈ sum(n .* σ̃²_pooled) / sum(n)^2 rtol = 1e-12
end

@testset "solve_ols_vcov: nᵢ - kᵢ matches entity-by-entity OLS" begin
    # With entity-owned columns, the pooled sandwich is each entity's textbook
    # s²(Xᵢ'Xᵢ)⁻¹, where s² = RSSᵢ/(nᵢ - kᵢ) and σu²vec carries RSSᵢ/(nᵢ - 1).
    Random.seed!(20261103)
    present = [1:30, 6:40, 11:24]
    df, obs_index = _unbalanced_index(present, 40)
    ids = Int.(levelcode.(df.id))
    time = df.t ./ 40
    X = reduce(hcat, [[ids .== i, (ids .== i) .* time, (ids .== i) .* time .^ 2][j] for i in 1:3 for j in 1:3])
    X = Float64.(X)
    y = randn(length(ids))
    u = y - X * (X \ y)
    n = length.(present)
    RSS = [sum(abs2, u[ids.==i]) for i in 1:3]
    textbook = zeros(9, 9)
    for i in 1:3
        cols = (3i-2):3i
        textbook[cols, cols] = RSS[i] / (n[i] - 3) * inv(X[ids.==i, cols]' * X[ids.==i, cols])
    end
    @test solve_ols_vcov(calculate_entity_variance(u, obs_index), X, obs_index) ≈ textbook rtol = 1e-10
end

@testset "solve_ols_vcov: balanced panels match common-T up to the dof factor" begin
    Random.seed!(20261031)
    N, T = 4, 24
    df, obs_index = _unbalanced_index(fill(1:T, N), T)
    σu² = 0.5 .+ rand(N)
    ids = Int.(levelcode.(df.id))
    η = randn(T)[df.t]
    post = df.t .> T ÷ 2
    entity_dummies = [Float64(ids[r] == i) for r in eachindex(ids), i in 1:N]
    X = hcat([(ids .== i) .* w for i in 1:N, w in (.!post, post, η .* .!post, η .* post)]...)
    @test solve_ols_vcov(σu², X, obs_index) ≈
          _common_t_ols_vcov(σu² .* (T - 1) / (T - 4), X, obs_index) rtol = 1e-12
    @test solve_ols_vcov(σu², entity_dummies, obs_index) ≈ Diagonal(σu² ./ T) rtol = 1e-12
end

@testset "solve_ols_vcov: rank and exact-fit handling" begin
    Random.seed!(20261104)
    z = randn(60)
    @test full_column_rank([60.0 sum(z); sum(z) dot(z, z)] .* 1e8)
    collinear = [ones(60) z 1e6 .+ 3e6 .* z]
    @test !full_column_rank(collinear' * collinear)
    @test !full_column_rank(zeros(2, 2))
    well_scaled = [ones(60) z 1e9 .* z .^ 2]
    @test full_column_rank(well_scaled' * well_scaled)

    # The last pivot of a collinear design is rounding noise of either sign, so one seed
    # proves nothing about the tolerance.
    misread = count(1:500) do seed
        Random.seed!(seed)
        w = randn(60)
        design = [ones(60) w 1 .+ 2 .* w]
        full_column_rank(design' * design)
    end
    @test misread == 0

    # A rank-deficient X skips leverage: no error, whatever the rounding in the inverse.
    # An exact-zero pivot raises `SingularException`.
    Random.seed!(20261104)
    present = [1:30, 6:40, 11:24]
    df, obs_index = _unbalanced_index(present, 40)
    ids = Int.(levelcode.(df.id))
    w = randn(length(ids))
    X = reduce(hcat, [[ids .== i, (ids .== i) .* w, (ids .== i) .* (0.1 .+ 0.3 .* w)][j]
                      for i in 1:3 for j in 1:3])
    X = Float64.(X)
    σu² = [0.5, 1.0, 2.0]
    @test !full_column_rank(X'X)
    outcome = try
        solve_ols_vcov(σu², X, obs_index)
    catch err
        err
    end
    @test outcome isa LinearAlgebra.SingularException || size(outcome) == (9, 9)

    # Full rank with an entity spanned by its own columns: that entity keeps σᵢ².
    present = [1:20, 1:3]
    df, obs_index = _unbalanced_index(present, 20)
    ids = Int.(levelcode.(df.id))
    exact = Float64.([ids .== 1 ids .== 2 (ids .== 2) .* df.t (ids .== 2) .* df.t .^ 2])
    @test full_column_rank(exact'exact)
    expected = zeros(4, 4)
    expected[1, 1] = 0.5 / 20   # entity 1: n = 20, k = 1, so the factor is one
    expected[2:4, 2:4] = 2.0 * inv(exact[ids.==2, 2:4]' * exact[ids.==2, 2:4])
    @test solve_ols_vcov([0.5, 2.0], exact, obs_index) ≈ expected rtol = 1e-8
end

@testset "giv: a collinear X keeps its point estimates" begin
    df = _load_simdata1()
    df.z = df.η1 .+ 0.25
    df.y = 3.0 .* df.z .- 2.0
    fit(f) = giv(df, f, :id, :t, :absS; guess=fill(1.0, 5), quiet=true, complete_coverage=true)
    m = fit(@formula(q + id & endog(p) ~ 0 + id + id & z + id & y))
    @test m.converged
    @test endog_coef(m) ≈ endog_coef(fit(@formula(q + id & endog(p) ~ 0 + id + id & z))) rtol = 1e-6
end

@testset "the degrees-of-freedom correction leaves estimation alone" begin
    df = _load_simdata1()
    m = giv(df, @formula(q + id & endog(p) ~ 0 + id & (η1 + η2)), :id, :t, :absS;
        guess=fill(1.0, 5), quiet=true, complete_coverage=true, save=:residuals)
    # Pinned `nᵢ - 1` results: the correction in `solve_ols_vcov` must not move them.
    @test endog_coef(m) ≈ [1.596360127433897, 1.6570008685549518, 1.2964302663478515,
        3.334970057944897, 0.5844260185011083] rtol = 1e-12
    @test exog_coef(m)[1:3] ≈ [0.34058483707463594, 0.3010093760396093,
        -1.3124893472735888] rtol = 1e-12
    ids = Int.(levelcode.(m.residual_df.id))
    residuals = m.residual_df.q_residual
    @test m.residual_variance ≈ [sum(abs2, residuals[ids.==i]) / (count(==(i), ids) - 1)
                                 for i in 1:5] rtol = 1e-12
end

@testset "giv: absorbed effects reach the OLS covariance through X_feres" begin
    df = _load_simdata1()
    m = giv(df, @formula(q + id & endog(p) ~ fe(id) + η1 + η2), :id, :t, :absS;
        guess=fill(1.0, 5), quiet=true, complete_coverage=true)
    ids = Int.(levelcode.(df.id))
    within(x) = x .- [mean(x[ids.==i]) for i in ids]
    X = hcat(within(df.η1), within(df.η2))
    Cp = reduce(hcat, [within(df.p .* (ids .== i)) for i in 1:5])
    XX = X'X
    k = [tr(XX \ (X[ids.==i, :]' * X[ids.==i, :])) for i in 1:5]
    T = length(unique(df.t))
    meat = sum(m.residual_variance[i] * (T - 1) / (T - k[i]) * X[ids.==i, :]' * X[ids.==i, :] for i in 1:5)
    β_Cp = X \ Cp
    @test exog_vcov(m) ≈ XX \ meat / XX + β_Cp * endog_vcov(m) * β_Cp' rtol = 1e-6
end

@testset "information formula matches its closed form" begin
    # Constant sizes with one elasticity per entity.
    Random.seed!(20261101)
    N, T = 4, 30
    df, obs_index = _unbalanced_index(fill(1:T, N), T)
    S = [0.4, 0.3, 0.2, 0.1]
    ζ = [0.5, 1.0, 1.5, 2.0]
    ids = Int.(levelcode.(df.id))
    u = randn(N * T)
    σu² = [sum(abs2, u[ids.==i]) / (T - 1) for i in 1:N]
    C = [Float64(ids[r] == i) for r in eachindex(ids), i in 1:N]
    _, Σζ = solve_optimal_vcov(ζ, u, S[ids], C, obs_index)
    Σ_S = sum(S .^ 2 .* σu²)
    information = [k == l ? (Σ_S / σu²[k] - S[k]^2) : S[k] * S[l] for k in 1:N, l in 1:N]
    @test Σζ ≈ dot(S, ζ)^2 * inv(information) / T rtol = 1e-10
end

@testset "giv: loading covariance on an unbalanced panel" begin
    Random.seed!(20261102)
    N, T = 4, 120
    ζ = [0.5, 1.0, 1.5, 2.0]
    σ = [0.6, 0.8, 1.0, 1.2]
    rows = DataFrame()
    for t in 1:T
        present = t <= T ÷ 2 ? (1:N) : (1:N-1)   # entity 4 leaves halfway
        S = fill(1 / length(present), length(present))
        u = σ[present] .* randn(length(present))
        p = dot(S, u) / dot(S, ζ[present])
        append!(rows, DataFrame(id=collect(present), t=t, S=S, p=p,
            q=u .- ζ[present] .* p))
    end
    rows.id = CategoricalArray(rows.id)
    m = giv(rows, @formula(q + id & endog(p) ~ 0 + id), :id, :t, :S;
        guess=ones(N), quiet=true, complete_coverage=true, save=:residuals)
    @test m.vcov_method === :optimal

    # Var(β̂ᵢ) = σᵢ²/nᵢ + p̄ᵢ² Var(ζ̂ᵢ) by hand: the first-step slope of p Cᵢ on entity i's
    # intercept is i's own mean price, and entity variances come from the saved residuals.
    residuals = m.residual_df.q_residual
    ids = Int.(levelcode.(m.residual_df.id))
    n = [count(==(i), ids) for i in 1:N]
    @test n == [T, T, T, T ÷ 2]
    σ̂² = [sum(abs2, residuals[ids.==i]) / (n[i] - 1) for i in 1:N]
    p̄ = [mean(rows.p[Int.(levelcode.(rows.id)).==i]) for i in 1:N]
    @test diag(exog_vcov(m)) ≈ σ̂² ./ n .+ p̄ .^ 2 .* diag(endog_vcov(m)) rtol = 1e-8
end
