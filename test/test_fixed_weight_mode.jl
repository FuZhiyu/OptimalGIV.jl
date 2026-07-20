using Test, OptimalGIV, Random, LinearAlgebra
using OptimalGIV: build_error_function, resolve_precision, calculate_entity_variance,
    solve_vcov, moment_conditions
using DataFrames, CSV, CategoricalArrays

# ---------------------------------------------------------------------------
# Fixed proxy / user precision weighting mode (precision_mode = :proxy | :fixed)
#
# Central claim (from giv-solver-stability/fixed-weight-mode): freezing the entity
# precisions makes the moment map an EXACT fixed quadratic in ζ under incomplete
# coverage (no period Mweights), removing the CUE self-weighting instability.
# ---------------------------------------------------------------------------

const _SIMDATA1 = joinpath(@__DIR__, "..", "examples", "simdata1.csv")

function _load_simdata1()
    df = CSV.read(_SIMDATA1, DataFrame)
    df.id = CategoricalArray(df.id)
    return df
end

# numerical equivalence between :iv and :iv_twopass only holds when the endogenous
# variable is excluded from the instrument (fe(id) & (η1+η2), no id&η)
const _FEQ = @formula(q + id & endog(p) ~ fe(id) & (η1 + η2) + 0)

@testset "fixed-weight mode: precision resolution" begin
    df = _load_simdata1()
    _, mats = build_error_function(df, _FEQ, :id, :t, :absS; algorithm=:iv, precision_mode=:proxy)
    obs_index = mats.obs_index
    N = obs_index.N

    # :cue resolves to `nothing` (keep continuously-updated weights)
    @test resolve_precision(:cue, nothing, mats.uq, obs_index) === nothing

    # :proxy is exactly 1/var(uq) from the residualized flows
    prox = resolve_precision(:proxy, nothing, mats.uq, obs_index)
    @test prox ≈ 1 ./ calculate_entity_variance(mats.uq, obs_index)
    @test length(prox) == N
    @test all(isfinite, prox)

    # :fixed round-trips the user-supplied vector
    w = collect(1.0:N)
    @test resolve_precision(:fixed, w, mats.uq, obs_index) ≈ w

    # error handling
    @test_throws ArgumentError resolve_precision(:fixed, nothing, mats.uq, obs_index)
    @test_throws ArgumentError resolve_precision(:fixed, ones(N + 1), mats.uq, obs_index)
    @test_throws ArgumentError resolve_precision(:nonsense, nothing, mats.uq, obs_index)
end

@testset "fixed-weight mode: :iv ≡ :iv_twopass" begin
    df = _load_simdata1()
    for mode in (:proxy, :fixed)
        # for :fixed, build the same 1/var(uq) proxy vector explicitly
        _, mats = build_error_function(df, _FEQ, :id, :t, :absS; algorithm=:iv, precision_mode=:proxy)
        w = mats.precision
        kw = mode == :fixed ? (; precision_mode=:fixed, precision_weights=w) : (; precision_mode=:proxy)

        m_iv = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true, algorithm=:iv, kw...)
        m_2p = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true, algorithm=:iv_twopass, kw...)
        @test maximum(abs, endog_coef(m_iv) - endog_coef(m_2p)) < 1e-8
        @test maximum(abs, vcov(m_iv) - vcov(m_2p)) < 1e-6
    end
end

@testset "fixed-weight mode: :fixed with proxy weights ≡ :proxy" begin
    df = _load_simdata1()
    _, mats = build_error_function(df, _FEQ, :id, :t, :absS; algorithm=:iv, precision_mode=:proxy)
    w = mats.precision
    m_proxy = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true, algorithm=:iv, precision_mode=:proxy)
    m_fixed = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true, algorithm=:iv,
        precision_mode=:fixed, precision_weights=w)
    @test endog_coef(m_proxy) ≈ endog_coef(m_fixed)
    @test vcov(m_proxy) ≈ vcov(m_fixed)
end

@testset "fixed-weight mode: proxy roots ≈ CUE roots (well-behaved fixture)" begin
    df = _load_simdata1()
    m_cue = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true, algorithm=:iv)
    m_proxy = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true, algorithm=:iv, precision_mode=:proxy)
    # both converge; proxy overweights via var(uq) ≥ var(u), so a small but nonzero gap
    @test m_cue.converged && m_proxy.converged
    @test maximum(abs, endog_coef(m_cue) - endog_coef(m_proxy)) < 1e-3
end

@testset "fixed-weight mode: moment map is exactly quadratic in ζ (incomplete coverage)" begin
    df = _load_simdata1()
    # force incomplete coverage so there are no period Mweights
    ef_proxy, _ = build_error_function(df, _FEQ, :id, :t, :absS;
        algorithm=:iv, complete_coverage=false, precision_mode=:proxy)
    ef_cue, _ = build_error_function(df, _FEQ, :id, :t, :absS;
        algorithm=:iv, complete_coverage=false, precision_mode=:cue)

    # third forward difference of a quadratic map vanishes identically:
    # Δ³g = g(x+3hd) - 3 g(x+2hd) + 3 g(x+hd) - g(x)
    third_diff(ef, x, d, h) = ef(x + 3h * d) - 3 * ef(x + 2h * d) + 3 * ef(x + h * d) - ef(x)

    Random.seed!(20240720)
    Nmom = 5
    proxy_norms = Float64[]
    cue_norms = Float64[]
    for _ in 1:5
        x = randn(Nmom)
        d = randn(Nmom)
        push!(proxy_norms, norm(third_diff(ef_proxy, x, d, 0.1)))
        push!(cue_norms, norm(third_diff(ef_cue, x, d, 0.1)))
    end
    # proxy: quadratic ⇒ third difference at machine precision
    @test maximum(proxy_norms) < 1e-9
    # teeth: CUE self-weighting is genuinely non-quadratic
    @test maximum(cue_norms) > 1e-6
end

@testset "fixed-weight mode: autodiff=:forward matches :central" begin
    df = _load_simdata1()
    for kw in ((; precision_mode=:proxy), (;))  # proxy and CUE default
        m_central = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true, algorithm=:iv,
            autodiff=:central, kw...)
        m_forward = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true, algorithm=:iv,
            autodiff=:forward, kw...)
        @test maximum(abs, endog_coef(m_central) - endog_coef(m_forward)) < 1e-8
    end
end

@testset "fixed-weight mode: vcov uses fixed weights, not CUE-optimal" begin
    df = _load_simdata1()
    m_proxy = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true, algorithm=:iv, precision_mode=:proxy)
    # reconstruct the sandwich vcov with the same fixed precision from the raw pieces
    _, mats = build_error_function(df, _FEQ, :id, :t, :absS; algorithm=:iv, precision_mode=:proxy)
    û = mats.uq + mats.uCp * endog_coef(m_proxy)
    _, Σref = solve_vcov(û, mats.S, mats.C, mats.uCp, mats.obs_index; precision=mats.precision)
    @test vcov(m_proxy) ≈ Σref
    @test all(isfinite, vcov(m_proxy))
    @test isposdef(Symmetric(vcov(m_proxy)))
end

@testset "fixed-weight mode: default (:cue) behavior unchanged" begin
    df = _load_simdata1()
    m_default = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true, algorithm=:iv)
    m_explicit_cue = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true, algorithm=:iv,
        precision_mode=:cue, method=:trust_region, autodiff=:central)
    @test endog_coef(m_default) == endog_coef(m_explicit_cue)
    @test vcov(m_default) == vcov(m_explicit_cue)
end
