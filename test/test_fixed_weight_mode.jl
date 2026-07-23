using Test, OptimalGIV, Random, LinearAlgebra, Logging, Statistics
using OptimalGIV: build_error_function, resolve_precision, calculate_entity_variance,
    aggregate_elasticity_in_domain, check_market_clearing, create_observation_index,
    estimate_giv, solve_vcov, moment_conditions, period_mweights
using DataFrames, CSV, CategoricalArrays

# ---------------------------------------------------------------------------
# Fixed raw/user precision weighting
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
    _, mats = build_error_function(df, _FEQ, :id, :t, :absS; algorithm=:iv, complete_coverage=true, precision_weights=:raw_onestep)
    obs_index = mats.obs_index
    N = obs_index.N

    # :cue resolves to `nothing` (keep continuously-updated weights)
    @test resolve_precision(:cue, mats.uq, obs_index) === nothing

    # :raw_onestep is exactly 1/var(uq) from the residualized flows
    prox = resolve_precision(:raw_onestep, mats.uq, obs_index)
    @test prox ≈ 1 ./ calculate_entity_variance(mats.uq, obs_index)
    @test length(prox) == N
    @test all(isfinite, prox)

    # :twostep resolves to the step-1 raw weights (giv() runs step 2 itself)
    @test resolve_precision(:twostep, mats.uq, obs_index) == prox

    # A vector round-trips as user-supplied fixed weights
    w = collect(1.0:N)
    @test resolve_precision(w, mats.uq, obs_index) ≈ w

    # error handling
    @test_throws ArgumentError resolve_precision(ones(N + 1), mats.uq, obs_index)
    @test_throws ArgumentError resolve_precision(:nonsense, mats.uq, obs_index)
end

@testset "coverage regime is explicit and validated" begin
    df = _load_simdata1()
    @test_throws UndefKeywordError giv(df, _FEQ, :id, :t, :absS;
        guess=ones(5), quiet=true, algorithm=:iv)
    @test_throws UndefKeywordError build_error_function(df, _FEQ, :id, :t, :absS;
        algorithm=:iv)

    _, mats = build_error_function(df, _FEQ, :id, :t, :absS;
        algorithm=:iv, complete_coverage=true, precision_weights=:raw_onestep)
    @test check_market_clearing(df.q, df.absS, mats.obs_index)
    @test_throws UndefKeywordError estimate_giv(mats.uq, mats.uCp, mats.C, mats.S,
        mats.obs_index, Val(:iv); guess=ones(5), quiet=true, precision=mats.precision)

    # An explicitly incomplete regime remains valid even when the realized sample adds up.
    m_incomplete = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true,
        algorithm=:iv, complete_coverage=false, precision_weights=:raw_onestep)
    @test !m_incomplete.complete_coverage

    # An asserted complete regime is a validation claim, not an override.
    df_bad = copy(df)
    df_bad.q[1] += 1.0
    @test_throws ArgumentError giv(df_bad, _FEQ, :id, :t, :absS; guess=ones(5),
        quiet=true, algorithm=:iv, complete_coverage=true, precision_weights=:raw_onestep)

    # Full-market-only algorithms reject incomplete coverage independently of logging.
    @test_throws ArgumentError giv(df, _FEQ, :id, :t, :absS; guess=ones(5),
        quiet=true, algorithm=:debiased_ols, complete_coverage=false, precision_weights=:cue)
    @test_throws ArgumentError giv(df, @formula(q + endog(p) ~ fe(id) & (η1 + η2) + 0),
        :id, :t, :absS; guess=Dict("Aggregate" => 2.0), quiet=true,
        algorithm=:scalar_search, complete_coverage=false)
end

@testset "algorithm × coverage × precision compatibility matrix" begin
    df = _load_simdata1()
    _, mats = build_error_function(df, _FEQ, :id, :t, :absS;
        algorithm=:iv, complete_coverage=true, precision_weights=:raw_onestep)
    custom = copy(mats.precision)
    precision_cases = [
        (:raw_onestep, :raw_onestep),
        (:twostep, :twostep),
        (:cue, :cue),
        (:custom, custom),
    ]
    iv_cases = [
        (; algorithm, coverage, precision_label, precision_weights)
        for algorithm in (:iv, :iv_twopass), coverage in (false, true),
            (precision_label, precision_weights) in precision_cases
    ]
    @test length(iv_cases) == 16
    for case in iv_cases
        @testset "$(case.algorithm), coverage=$(case.coverage), $(case.precision_label)" begin
            m = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true,
                algorithm=case.algorithm, complete_coverage=case.coverage,
                precision_weights=case.precision_weights, return_vcov=false)
            @test m.complete_coverage == case.coverage
            @test m.converged
            @test all(isfinite, endog_coef(m))
        end
    end

    full_market_formula = @formula(q + endog(p) ~ 0 + fe(id) & (η1 + η2))
    specialized_cases = [
        (; algorithm=:debiased_ols, guess=[1.0], precision_weights=:cue),
        (; algorithm=:scalar_search, guess=Dict("Aggregate" => 2.0), precision_weights=nothing),
    ]
    for case in specialized_cases
        @testset "$(case.algorithm), complete coverage" begin
            kwargs = isnothing(case.precision_weights) ? (;) :
                     (; precision_weights=case.precision_weights)
            m = giv(df, full_market_formula, :id, :t, :absS; guess=case.guess,
                quiet=true, algorithm=case.algorithm, complete_coverage=true,
                return_vcov=false, kwargs...)
            @test m.complete_coverage
            @test m.converged
        end
        @testset "$(case.algorithm), incomplete coverage rejected quietly" begin
            kwargs = isnothing(case.precision_weights) ? (;) :
                     (; precision_weights=case.precision_weights)
            @test_throws ArgumentError giv(df, full_market_formula, :id, :t, :absS;
                guess=case.guess, quiet=true, algorithm=case.algorithm,
                complete_coverage=false, return_vcov=false, kwargs...)
        end
    end
end

@testset "complete-coverage clamp and final-root domain" begin
    panel = DataFrame(id=CategoricalArray(repeat(1:2, 4)), t=repeat(1:4; inner=2))
    obs_index = create_observation_index(panel, :id, :t, Dict{Int,Vector{Int}}())
    q = repeat([1.0, 2.0], 4)
    Cp = ones(8, 1)
    C = ones(8, 1)
    S = fill(0.5, 8)
    Mw = period_mweights([0.0], C, S, obs_index)
    @test all(isfinite, Mw) && sum(Mw) == 1.0
    @test !aggregate_elasticity_in_domain([0.0], C, S, obs_index)
    @test aggregate_elasticity_in_domain([1.0], C, S, obs_index)

    ζ̂, converged = estimate_giv(q, Cp, C, S, obs_index, Val(:iv);
        guess=[-1.0], quiet=true, complete_coverage=true,
        precision=ones(2), solver_options=(; ftol=1e-10))
    @test ζ̂ == [-1.0]
    @test !converged
end

@testset "fixed-weight mode: :iv ≡ :iv_twopass" begin
    df = _load_simdata1()
    for mode in (:raw_onestep, :custom)
        # For :custom, build the same 1/var(uq) raw vector explicitly
        _, mats = build_error_function(df, _FEQ, :id, :t, :absS; algorithm=:iv, complete_coverage=true, precision_weights=:raw_onestep)
        w = mats.precision
        kw = mode == :custom ? (; precision_weights=w) : (; precision_weights=:raw_onestep)

        m_iv = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true, algorithm=:iv, complete_coverage=true, kw...)
        m_2p = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true, algorithm=:iv_twopass, complete_coverage=true, kw...)
        @test maximum(abs, endog_coef(m_iv) - endog_coef(m_2p)) < 1e-8
        @test maximum(abs, vcov(m_iv) - vcov(m_2p)) < 1e-6
    end
end

@testset "fixed-weight mode: custom raw weights ≡ :raw_onestep" begin
    df = _load_simdata1()
    _, mats = build_error_function(df, _FEQ, :id, :t, :absS; algorithm=:iv, complete_coverage=true, precision_weights=:raw_onestep)
    w = mats.precision
    m_proxy = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true, algorithm=:iv, complete_coverage=true, precision_weights=:raw_onestep)
    m_fixed = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true, algorithm=:iv,
        precision_weights=w, complete_coverage=true)
    @test endog_coef(m_proxy) ≈ endog_coef(m_fixed)
    @test vcov(m_proxy) ≈ vcov(m_fixed)
end

@testset "fixed-weight mode: raw one-step roots ≈ CUE roots (well-behaved fixture)" begin
    df = _load_simdata1()
    m_cue = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true, algorithm=:iv,
        precision_weights=:cue, complete_coverage=true)
    m_proxy = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true, algorithm=:iv, complete_coverage=true, precision_weights=:raw_onestep)
    # Both converge; raw weights over-weight via var(uq) ≥ var(u), leaving a small gap.
    @test m_cue.converged && m_proxy.converged
    @test maximum(abs, endog_coef(m_cue) - endog_coef(m_proxy)) < 1e-3
end

@testset "fixed-weight mode: moment map is exactly quadratic in ζ (incomplete coverage)" begin
    df = _load_simdata1()
    # force incomplete coverage so there are no period Mweights
    ef_proxy, _ = build_error_function(df, _FEQ, :id, :t, :absS;
        algorithm=:iv, complete_coverage=false, precision_weights=:raw_onestep)
    ef_cue, _ = build_error_function(df, _FEQ, :id, :t, :absS;
        algorithm=:iv, complete_coverage=false, precision_weights=:cue)

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
    # raw one-step: quadratic ⇒ third difference at machine precision
    @test maximum(proxy_norms) < 1e-9
    # teeth: CUE self-weighting is genuinely non-quadratic
    @test maximum(cue_norms) > 1e-6
end

@testset "solver_options: method/autodiff route without keyword conflicts" begin
    df = _load_simdata1()
    base = (; ftol=1e-8, show_trace=false, iterations=100, method=:trust_region)
    for weights in (:raw_onestep, :cue)
        m_central = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true, algorithm=:iv,
            precision_weights=weights, complete_coverage=true, solver_options=(; base..., autodiff=:central))
        m_forward = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true, algorithm=:iv,
            precision_weights=weights, complete_coverage=true, solver_options=(; base..., autodiff=:forward))
        @test m_central.converged && m_forward.converged
        @test maximum(abs, endog_coef(m_central) - endog_coef(m_forward)) < 1e-8
    end
end

@testset "fixed-weight mode: vcov uses fixed weights, not CUE-optimal" begin
    df = _load_simdata1()
    m_proxy = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true, algorithm=:iv, complete_coverage=true, precision_weights=:raw_onestep)
    # reconstruct the sandwich vcov with the same fixed precision from the raw pieces
    _, mats = build_error_function(df, _FEQ, :id, :t, :absS; algorithm=:iv, complete_coverage=true, precision_weights=:raw_onestep)
    û = mats.uq + mats.uCp * endog_coef(m_proxy)
    _, Σref = solve_vcov(û, mats.S, mats.C, mats.uCp, mats.obs_index; precision=mats.precision)
    @test vcov(m_proxy) ≈ Σref
    @test all(isfinite, vcov(m_proxy))
    @test isposdef(Symmetric(vcov(m_proxy)))
end

# ---------------------------------------------------------------------------
# Complete-coverage feasible two-step: raw/custom one-step uses equal periods;
# step 2 freezes period_mweights constructed at the step-1 root and carries the
# same vector into its moments, Jacobian, and sandwich vcov.
# ---------------------------------------------------------------------------

# Independent O(N²) reference: build the actual masked period scores and
# moment-by-parameter bread directly from the within-period pair loop, without
# using the production pair catalog or score constructor.
function _reference_sandwich_vcov(u, S, C, Cp, obs_index, prec, Mw)
    Nmom = size(C, 2)
    T = obs_index.T
    scores = zeros(Nmom, T)
    weightsum = zeros(Nmom, T)
    Gt = zeros(Nmom, Nmom, T)
    for t in 1:T
        r = obs_index.start_indices[t]:obs_index.end_indices[t]
        mw = isnothing(Mw) ? 1.0 : Mw[t]
        for ii in r, jj in (ii+1):last(r)
            i, j = obs_index.ids[ii], obs_index.ids[jj]
            obs_index.exclpairs[i, j] && continue
            w = [prec[i] * S[jj] * C[ii, k] + prec[j] * S[ii] * C[jj, k] for k in 1:Nmom]
            d = [u[jj] * Cp[ii, k] + u[ii] * Cp[jj, k] for k in 1:Nmom]
            scores[:, t] .+= mw .* w .* (u[ii] * u[jj])
            weightsum[:, t] .+= w
            Gt[:, :, t] .+= mw .* (w * d')
        end
    end
    momweight = vec(sum(abs.(weightsum); dims=2))
    momweight ./= sum(momweight)
    scores ./= reshape(momweight, :, 1)
    Gt ./= reshape(momweight, :, 1, 1)
    G = dropdims(mean(Gt; dims=3); dims=3)
    centered_scores = scores .- mean(scores; dims=2)
    B = centered_scores * centered_scores' / T
    invG = inv(G)
    Σ = invG * B * invG' / T
    return Symmetric(Σ + Σ') / 2
end

function _timevarying_complete_fixture()
    # DGP with genuinely time-varying aggregate elasticity: sizes S_it move over t
    # and p_t clears the market exactly.
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
    return df, fml, ζtrue
end

@testset "complete-coverage two-step: frozen period-scaled moments and sandwich" begin
    df, fml, ζtrue = _timevarying_complete_fixture()

    m1 = giv(df, fml, :id, :t, :S; guess=ζtrue, quiet=true, algorithm=:iv,
        complete_coverage=true, precision_weights=:raw_onestep)
    @test m1.complete_coverage && m1.converged
    ef_complete, mats = build_error_function(df, fml, :id, :t, :S;
        algorithm=:iv, complete_coverage=true, precision_weights=:raw_onestep)
    ef_incomplete, _ = build_error_function(df, fml, :id, :t, :S;
        algorithm=:iv, complete_coverage=false, precision_weights=:raw_onestep)
    @test ef_complete(ζtrue) == ef_incomplete(ζtrue) # raw one-step uses equal periods

    û₁ = mats.uq + mats.uCp * endog_coef(m1)
    w2 = 1 ./ calculate_entity_variance(û₁, mats.obs_index)
    Mw1 = period_mweights(endog_coef(m1), mats.C, mats.S, mats.obs_index)
    @test maximum(Mw1) / minimum(Mw1) > 1.2

    solver = (; ftol=1e-6, show_trace=false, iterations=100)
    ζ2, converged2 = estimate_giv(mats.uq, mats.uCp, mats.C, mats.S, mats.obs_index,
        Val(:iv); guess=endog_coef(m1), quiet=true, complete_coverage=true,
        solver_options=solver, precision=w2, Mweights=Mw1)
    m2 = giv(df, fml, :id, :t, :S; guess=ζtrue, quiet=true, algorithm=:iv,
        complete_coverage=true, precision_weights=:twostep)
    @test converged2 && m2.converged
    @test endog_coef(m2) == ζ2

    û₂ = mats.uq + mats.uCp * ζ2
    _, Σ_scaled = solve_vcov(û₂, mats.S, mats.C, mats.uCp, mats.obs_index;
        precision=w2, Mweights=Mw1)
    @test vcov(m2) == Σ_scaled

    # Teeth: recomputing period weights at the second-step root changes the vcov.
    Mw2 = period_mweights(ζ2, mats.C, mats.S, mats.obs_index)
    _, Σ_recomputed = solve_vcov(û₂, mats.S, mats.C, mats.uCp, mats.obs_index;
        precision=w2, Mweights=Mw2)
    @test norm(Σ_scaled - Σ_recomputed) / norm(Σ_scaled) > 1e-6
    # independent O(N²) reference reproduces the fast implementation
    Σ_ref = _reference_sandwich_vcov(û₂, mats.S, mats.C, mats.uCp, mats.obs_index,
        w2, Mw1)
    @test Σ_scaled ≈ Σ_ref rtol = 1e-8
end

@testset "complete-coverage vcov: uniform Mweights leave simdata1 SEs unchanged" begin
    # simdata1's aggregate elasticity is constant over time ⇒ Mweights are uniform,
    # and a uniform rescaling of the periods cancels in A⁻¹BA⁻ᵀ: the fix must leave
    # these SEs unchanged relative to the unscaled sandwich.
    df = _load_simdata1()
    m = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true, algorithm=:iv,
        complete_coverage=true, precision_weights=:raw_onestep)
    @test m.complete_coverage

    _, mats = build_error_function(df, _FEQ, :id, :t, :absS;
        algorithm=:iv, complete_coverage=true, precision_weights=:raw_onestep)
    û = mats.uq + mats.uCp * endog_coef(m)
    Mw = period_mweights(endog_coef(m), mats.C, mats.S, mats.obs_index)
    @test maximum(Mw) - minimum(Mw) < 1e-12   # uniform period weights

    _, Σ_unscaled = solve_vcov(û, mats.S, mats.C, mats.uCp, mats.obs_index; precision=mats.precision)
    @test vcov(m) ≈ Σ_unscaled rtol = 1e-10
end

@testset "complete-coverage vcov: incomplete-coverage fixed-mode SEs byte-unchanged" begin
    df = _load_simdata1()
    m = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true, algorithm=:iv,
        precision_weights=:raw_onestep, complete_coverage=false)
    _, mats = build_error_function(df, _FEQ, :id, :t, :absS; algorithm=:iv,
        precision_weights=:raw_onestep, complete_coverage=false)
    û = mats.uq + mats.uCp * endog_coef(m)
    _, Σ = solve_vcov(û, mats.S, mats.C, mats.uCp, mats.obs_index; precision=mats.precision)
    @test vcov(m) == Σ   # byte-identical: no Mweights anywhere in the incomplete path
end

@testset "two-step frozen weights compose with pin_zero" begin
    df, fml, ζtrue = _timevarying_complete_fixture()
    _, _, names, _, _ = OptimalGIV.get_coefnames(df, fml)
    pin = names[1]
    m1 = giv(df, fml, :id, :t, :S; guess=ζtrue, quiet=true,
        algorithm=:iv, complete_coverage=true, precision_weights=:raw_onestep,
        pin_zero=[pin])
    m = giv(df, fml, :id, :t, :S; guess=ζtrue, quiet=true,
        algorithm=:iv, complete_coverage=true, precision_weights=:twostep,
        pin_zero=[pin])
    @test m1.converged && m.converged
    @test endog_coef(m)[1] == 0.0
    @test all(iszero, vcov(m)[1, :])
    @test all(iszero, vcov(m)[:, 1])
    @test all(isfinite, vcov(m)[2:end, 2:end])

    ef, mats = build_error_function(df, fml, :id, :t, :S;
        algorithm=:iv, complete_coverage=true, precision_weights=:twostep,
        pin_zero=[pin])
    @test size(mats.C, 2) == 4
    @test length(ef(ones(4))) == 4

    ζ1_free = endog_coef(m1)[2:end]
    ζ2_free = endog_coef(m)[2:end]
    û1 = mats.uq + mats.uCp * ζ1_free
    w2 = 1 ./ calculate_entity_variance(û1, mats.obs_index)
    Mw1 = period_mweights(ζ1_free, mats.C, mats.S, mats.obs_index)
    @test maximum(Mw1) / minimum(Mw1) > 1.2

    solver = (; ftol=1e-6, show_trace=false, iterations=100)
    ζ2_direct, converged2 = estimate_giv(mats.uq, mats.uCp, mats.C, mats.S,
        mats.obs_index, Val(:iv); guess=ζ1_free, quiet=true, complete_coverage=true,
        solver_options=solver, precision=w2, Mweights=Mw1)
    @test converged2 && ζ2_free == ζ2_direct

    û2 = mats.uq + mats.uCp * ζ2_free
    _, Σ_frozen = solve_vcov(û2, mats.S, mats.C, mats.uCp, mats.obs_index;
        precision=w2, Mweights=Mw1)
    @test vcov(m)[2:end, 2:end] == Σ_frozen

    Mw2 = period_mweights(ζ2_free, mats.C, mats.S, mats.obs_index)
    _, Σ_recomputed = solve_vcov(û2, mats.S, mats.C, mats.uCp, mats.obs_index;
        precision=w2, Mweights=Mw2)
    @test norm(Σ_frozen - Σ_recomputed) / norm(Σ_frozen) > 1e-4
end

# ---------------------------------------------------------------------------
# Two-step default (precision_weights = :twostep; giv-solver-stability/twostep-default)
#
# The package default estimator is now the two-step efficient GMM: step 1 solves
# with :raw_onestep weights, step 2 re-solves once with precisions 1/var(û₁ᵢ) from the
# step-1 residuals, warm-started at the step-1 root.
# ---------------------------------------------------------------------------

@testset "twostep default: implicit default ≡ explicit :twostep (bit-equal)" begin
    df = _load_simdata1()
    m_default = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true, algorithm=:iv, complete_coverage=true)
    m_twostep = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true, algorithm=:iv,
        precision_weights=:twostep, complete_coverage=true)
    @test endog_coef(m_default) == endog_coef(m_twostep)
    @test vcov(m_default) == vcov(m_twostep)
    @test m_default.converged
    # :cue remains available opt-in and is a genuinely different estimator here
    m_cue = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true, algorithm=:iv,
        precision_weights=:cue, complete_coverage=true)
    @test endog_coef(m_cue) != endog_coef(m_twostep)
    @test maximum(abs, endog_coef(m_cue) - endog_coef(m_twostep)) < 1e-3  # near-efficient
end

@testset "twostep default: silent :twostep resolution (shim removed)" begin
    df = _load_simdata1()
    gcall(; kw...) = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), algorithm=:iv,
        complete_coverage=true, solver_options=(; ftol=1e-6, show_trace=false, iterations=100), kw...)

    # Omitting `precision_weights` resolves to :twostep with no migration warning
    # and records the resolved mode on the model.
    @test_logs min_level=Logging.Warn gcall()
    @test_logs min_level=Logging.Warn gcall(precision_weights=:twostep)
    m_default = gcall()
    @test m_default.precision_weights === :twostep
    @test endog_coef(m_default) == endog_coef(gcall(precision_weights=:twostep))
end

@testset "twostep: step 2 ≡ custom step-1-residual precisions" begin
    df = _load_simdata1()
    # step 1 by hand: the :raw_onestep solve and its residuals
    m1 = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true, algorithm=:iv,
        precision_weights=:raw_onestep, complete_coverage=true)
    @test m1.converged
    _, mats = build_error_function(df, _FEQ, :id, :t, :absS; algorithm=:iv,
        complete_coverage=true, precision_weights=:raw_onestep)
    û₁ = mats.uq + mats.uCp * endog_coef(m1)
    w2 = 1 ./ calculate_entity_variance(û₁, mats.obs_index)
    # step 2 by hand: custom step-1-residual precisions, warm-started at ζ̂₁
    m_fixed = giv(df, _FEQ, :id, :t, :absS; guess=endog_coef(m1), quiet=true, algorithm=:iv,
        precision_weights=w2, complete_coverage=true)
    m_two = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true, algorithm=:iv,
        precision_weights=:twostep, complete_coverage=true)
    @test endog_coef(m_two) == endog_coef(m_fixed)
    @test vcov(m_two) ≈ vcov(m_fixed) rtol = 1e-12
end
