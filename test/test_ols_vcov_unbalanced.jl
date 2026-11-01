using Test, OptimalGIV, Random, LinearAlgebra, Statistics
using OptimalGIV: create_observation_index, solve_ols_vcov, solve_optimal_vcov
using DataFrames, CategoricalArrays

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
function _pre_fix_ols_vcov(σu², X, obs_index)
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
    @test diag(_pre_fix_ols_vcov(σu², entity_dummies, obs_index)) ≈ σu² ./ T rtol = 1e-12

    # Regime-split entity intercepts: σᵢ²/nᵢᵣ within each occupied regime.
    post = df.t .> 20
    regime_columns = [entity_dummies .* .!post entity_dummies .* post]
    occupied = vec(any(!iszero, regime_columns; dims=1))
    nᵢᵣ = vec(sum(regime_columns; dims=1))[occupied]
    @test diag(solve_ols_vcov(σu², regime_columns[:, occupied], obs_index)) ≈
          repeat(σu², 2)[occupied] ./ nᵢᵣ rtol = 1e-12

    # One pooled intercept: the variance of the mean of all rows.
    pooled = ones(length(ids), 1)
    @test only(solve_ols_vcov(σu², pooled, obs_index)) ≈ sum(n .* σu²) / sum(n)^2 rtol = 1e-12
end

@testset "solve_ols_vcov: balanced panels reproduce the common-T formula" begin
    Random.seed!(20261031)
    N, T = 4, 24
    df, obs_index = _unbalanced_index(fill(1:T, N), T)
    σu² = 0.5 .+ rand(N)
    ids = Int.(levelcode.(df.id))
    η = randn(T)[df.t]
    post = df.t .> T ÷ 2
    entity_dummies = [Float64(ids[r] == i) for r in eachindex(ids), i in 1:N]
    X = hcat([(ids .== i) .* w for i in 1:N, w in (.!post, post, η .* .!post, η .* post)]...)
    @test solve_ols_vcov(σu², X, obs_index) ≈ _pre_fix_ols_vcov(σu², X, obs_index) rtol = 1e-12
    @test solve_ols_vcov(σu², entity_dummies, obs_index) ≈ Diagonal(σu² ./ T) rtol = 1e-12
end

@testset "information formula matches the closed form of Theorem 1" begin
    # Constant sizes with one elasticity per entity (eq:asymptotic_variance_zeta).
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

    # eq:beta_variance by hand: the first-step slope of p Cᵢ on entity i's intercept
    # is i's own mean price, and entity variances come from the saved residuals.
    residuals = m.residual_df.q_residual
    ids = Int.(levelcode.(m.residual_df.id))
    n = [count(==(i), ids) for i in 1:N]
    @test n == [T, T, T, T ÷ 2]
    σ̂² = [sum(abs2, residuals[ids.==i]) / (n[i] - 1) for i in 1:N]
    p̄ = [mean(rows.p[Int.(levelcode.(rows.id)).==i]) for i in 1:N]
    @test diag(exog_vcov(m)) ≈ σ̂² ./ n .+ p̄ .^ 2 .* diag(endog_vcov(m)) rtol = 1e-8
end
