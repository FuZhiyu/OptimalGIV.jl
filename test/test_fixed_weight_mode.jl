using Test, OptimalGIV, Random, LinearAlgebra, Logging
using OptimalGIV: build_error_function, resolve_precision, calculate_entity_variance,
    solve_vcov, moment_conditions, period_mweights
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

    # :twostep resolves to the step-1 proxy weights (giv() runs step 2 itself)
    @test resolve_precision(:twostep, nothing, mats.uq, obs_index) == prox

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

# ---------------------------------------------------------------------------
# Complete-coverage sandwich SEs: the :iv kernels scale each period's moments by
# period_mweights (∝ 1/clamp(|ζS_t|, √eps, Inf), normalized); under the fixed-weight
# modes solve_vcov must carry the same Mweights (evaluated at the solution) into W
# so the SEs match the moments actually solved (giv-solver-stability/
# complete-coverage-vcov).
# ---------------------------------------------------------------------------

# Independent O(N²) reference: build the period-scaled sandwich A/B directly from
# the within-period pair loop, mirroring solve_vcov's math (same A/(T-1), B/T
# normalizations) without its pair-array machinery.
function _reference_sandwich_vcov(u, S, C, Cp, obs_index, prec, Mw)
    Nmom = size(C, 2)
    T = obs_index.T
    σu² = calculate_entity_variance(u, obs_index)
    A = zeros(Nmom, Nmom)
    B = zeros(Nmom, Nmom)
    for t in 1:T
        r = obs_index.start_indices[t]:obs_index.end_indices[t]
        mw = isnothing(Mw) ? 1.0 : Mw[t]
        for ii in r, jj in (ii+1):last(r)
            i, j = obs_index.ids[ii], obs_index.ids[jj]
            w = [mw * (prec[i] * S[jj] * C[ii, k] + prec[j] * S[ii] * C[jj, k]) for k in 1:Nmom]
            d = [u[jj] * Cp[ii, k] + u[ii] * Cp[jj, k] for k in 1:Nmom]
            A .+= d * w'                     # (D'W)[k, l] summed pair by pair
            B .+= (σu²[i] * σu²[j]) .* (w * w')  # (W' diag(V) W) summed pair by pair
        end
    end
    A ./= (T - 1)
    B ./= T
    B = Symmetric(B + B') / 2
    invA = inv(A)
    Σ = invA * B * invA' / T
    return Symmetric(Σ + Σ') / 2
end

@testset "complete-coverage vcov: period-scaled sandwich (time-varying ζS)" begin
    # DGP with genuinely time-varying aggregate elasticity: sizes S_it move over t
    # and p_t clears the market exactly, so complete coverage is auto-detected.
    Random.seed!(20260720)
    N, T = 5, 80
    ζtrue = [0.5, 1.0, 1.5, 2.0, 3.0]
    Smat = rand(N, T) .^ 3 .+ 0.05
    Smat ./= sum(Smat; dims=1)
    umat = randn(N, T) .* (0.5 .+ rand(N))
    p = [dot(Smat[:, t], umat[:, t]) / dot(Smat[:, t], ζtrue) for t in 1:T]
    qmat = umat .- ζtrue * p'
    df = DataFrame(
        id=CategoricalArray(repeat(1:N, T)),
        t=repeat(1:T; inner=N),
        q=vec(qmat),
        p=repeat(p; inner=N),
        S=vec(Smat),
    )
    fml = @formula(q + id & endog(p) ~ 0)

    m = giv(df, fml, :id, :t, :S; guess=ζtrue, quiet=true, algorithm=:iv, precision_mode=:proxy)
    @test m.complete_coverage
    @test m.converged

    _, mats = build_error_function(df, fml, :id, :t, :S; algorithm=:iv, precision_mode=:proxy)
    û = mats.uq + mats.uCp * endog_coef(m)
    Mw = period_mweights(endog_coef(m), mats.C, mats.S, mats.obs_index)
    @test maximum(Mw) / minimum(Mw) > 1.2   # period weights genuinely time-varying

    _, Σ_scaled = solve_vcov(û, mats.S, mats.C, mats.uCp, mats.obs_index;
        precision=mats.precision, Mweights=Mw)
    _, Σ_unscaled = solve_vcov(û, mats.S, mats.C, mats.uCp, mats.obs_index;
        precision=mats.precision)

    # giv() routes complete-coverage fixed-mode SEs through the period-scaled sandwich
    @test vcov(m) ≈ Σ_scaled
    # teeth: the period scaling genuinely moves the SEs
    @test norm(Σ_scaled - Σ_unscaled) / norm(Σ_unscaled) > 1e-3
    # independent O(N²) reference reproduces the fast implementation
    Σ_ref = _reference_sandwich_vcov(û, mats.S, mats.C, mats.uCp, mats.obs_index,
        mats.precision, Mw)
    @test Σ_scaled ≈ Σ_ref rtol = 1e-8
    # and with Mw === nothing the reference also reproduces the unscaled sandwich
    Σ_ref0 = _reference_sandwich_vcov(û, mats.S, mats.C, mats.uCp, mats.obs_index,
        mats.precision, nothing)
    @test Σ_unscaled ≈ Σ_ref0 rtol = 1e-8
end

@testset "complete-coverage vcov: uniform Mweights leave simdata1 SEs unchanged" begin
    # simdata1's aggregate elasticity is constant over time ⇒ Mweights are uniform,
    # and a uniform rescaling of the periods cancels in A⁻¹BA⁻ᵀ: the fix must leave
    # these SEs unchanged relative to the unscaled sandwich.
    df = _load_simdata1()
    m = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true, algorithm=:iv, precision_mode=:proxy)
    @test m.complete_coverage

    _, mats = build_error_function(df, _FEQ, :id, :t, :absS; algorithm=:iv, precision_mode=:proxy)
    û = mats.uq + mats.uCp * endog_coef(m)
    Mw = period_mweights(endog_coef(m), mats.C, mats.S, mats.obs_index)
    @test maximum(Mw) - minimum(Mw) < 1e-12   # uniform period weights

    _, Σ_unscaled = solve_vcov(û, mats.S, mats.C, mats.uCp, mats.obs_index; precision=mats.precision)
    @test vcov(m) ≈ Σ_unscaled rtol = 1e-10
end

@testset "complete-coverage vcov: incomplete-coverage fixed-mode SEs byte-unchanged" begin
    df = _load_simdata1()
    m = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true, algorithm=:iv,
        precision_mode=:proxy, complete_coverage=false)
    _, mats = build_error_function(df, _FEQ, :id, :t, :absS; algorithm=:iv,
        precision_mode=:proxy, complete_coverage=false)
    û = mats.uq + mats.uCp * endog_coef(m)
    _, Σ = solve_vcov(û, mats.S, mats.C, mats.uCp, mats.obs_index; precision=mats.precision)
    @test vcov(m) == Σ   # byte-identical: no Mweights anywhere in the incomplete path
end

# ---------------------------------------------------------------------------
# Two-step default (precision_mode = :twostep; giv-solver-stability/twostep-default)
#
# The package default estimator is now the two-step efficient GMM: step 1 solves
# with :proxy weights, step 2 re-solves once with precisions 1/var(û₁ᵢ) from the
# step-1 residuals, warm-started at the step-1 root. The `precision_mode = nothing`
# sentinel resolves to :twostep with a one-time behavior-change warning.
# ---------------------------------------------------------------------------

@testset "twostep default: implicit default ≡ explicit :twostep (bit-equal)" begin
    df = _load_simdata1()
    m_default = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true, algorithm=:iv)
    m_twostep = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true, algorithm=:iv,
        precision_mode=:twostep)
    @test endog_coef(m_default) == endog_coef(m_twostep)
    @test vcov(m_default) == vcov(m_twostep)
    @test m_default.converged
    # :cue remains available opt-in and is a genuinely different estimator here
    m_cue = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true, algorithm=:iv,
        precision_mode=:cue)
    @test endog_coef(m_cue) != endog_coef(m_twostep)
    @test maximum(abs, endog_coef(m_cue) - endog_coef(m_twostep)) < 1e-3  # near-efficient
end

@testset "twostep: step 2 ≡ :fixed with step-1-residual precisions" begin
    df = _load_simdata1()
    # step 1 by hand: the :proxy solve and its residuals
    m1 = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true, algorithm=:iv,
        precision_mode=:proxy)
    @test m1.converged
    _, mats = build_error_function(df, _FEQ, :id, :t, :absS; algorithm=:iv, precision_mode=:proxy)
    û₁ = mats.uq + mats.uCp * endog_coef(m1)
    w2 = 1 ./ calculate_entity_variance(û₁, mats.obs_index)
    # step 2 by hand: :fixed with the step-1-residual precisions, warm-started at ζ̂₁
    m_fixed = giv(df, _FEQ, :id, :t, :absS; guess=endog_coef(m1), quiet=true, algorithm=:iv,
        precision_mode=:fixed, precision_weights=w2)
    m_two = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true, algorithm=:iv,
        precision_mode=:twostep)
    @test endog_coef(m_two) == endog_coef(m_fixed)
    @test vcov(m_two) == vcov(m_fixed)
end

@testset "twostep default: one-time behavior-change warning" begin
    df = _load_simdata1()
    solveropts = (; ftol=1e-6, show_trace=false, iterations=100)
    gcall(; kw...) = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), algorithm=:iv,
        solver_options=solveropts, kw...)

    OptimalGIV._TWOSTEP_DEFAULT_WARNED[] = false
    # explicit modes never warn
    for mode in (:twostep, :cue, :proxy)
        @test_logs min_level = Logging.Warn gcall(precision_mode=mode)
    end
    @test !OptimalGIV._TWOSTEP_DEFAULT_WARNED[]
    # quiet=true silences the implicit default and does NOT consume the one-time warning
    @test_logs min_level = Logging.Warn gcall(quiet=true)
    @test !OptimalGIV._TWOSTEP_DEFAULT_WARNED[]
    # first non-quiet implicit call warns exactly once...
    @test_logs (:warn, r"default GIV estimator changed from :cue to :twostep") match_mode = :any gcall()
    @test OptimalGIV._TWOSTEP_DEFAULT_WARNED[]
    # ...and never again in the same session
    @test_logs min_level = Logging.Warn gcall()
    OptimalGIV._TWOSTEP_DEFAULT_WARNED[] = false  # leave a clean state for other suites
end
