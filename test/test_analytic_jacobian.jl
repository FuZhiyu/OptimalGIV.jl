using Test, OptimalGIV, Random, LinearAlgebra
using OptimalGIV: build_error_function, mean_moment_jacobian
using OptimalGIV.NLsolve: nlsolve, converged
using ForwardDiff
using DataFrames, CSV, CategoricalArrays

# ---------------------------------------------------------------------------
# Analytic Jacobian of the fixed-precision moment kernel
# (giv-solver-stability/analytic-jacobian)
#
# With fixed precisions the residuals u = uq + uCp ζ are linear in ζ, so the
# Jacobian of the :iv/:iv_twopass moment map is exact in closed form (same O(N)
# pair-sum identity as the moments; constant momweight; closed-form Mweights
# product-rule term under complete coverage). These tests pin the hand-coded
# Jacobian against ForwardDiff across coverage regimes and exclusion patterns,
# and check the solver root is unchanged relative to the FD/autodiff path.
# ---------------------------------------------------------------------------

const _SIMDATA1_AJ = joinpath(@__DIR__, "..", "examples", "simdata1.csv")

function _load_simdata1_aj()
    df = CSV.read(_SIMDATA1_AJ, DataFrame)
    df.id = CategoricalArray(df.id)
    return df
end

const _FEQ_AJ = @formula(q + id & endog(p) ~ fe(id) & (η1 + η2) + 0)

# DGP with genuinely time-varying aggregate elasticity ζS_t (sizes move over t,
# p_t clears the market exactly ⇒ complete coverage auto-detected); reused from
# the complete-coverage-vcov tests in test_fixed_weight_mode.jl.
function _tv_zetaS_df(; N=5, T=80, seed=20260720)
    Random.seed!(seed)
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
    return df, ζtrue
end

# analytic vs ForwardDiff at a set of ζ points (both exact ⇒ machine-precision agreement)
function _test_jac_vs_forwarddiff(err_func, jac_func, ζpoints)
    for ζ in ζpoints
        J_fd = ForwardDiff.jacobian(err_func, ζ)
        J_an = jac_func(ζ)
        @test maximum(abs, J_an - J_fd) < 1e-8 * max(1.0, maximum(abs, J_fd))
    end
end

@testset "analytic Jacobian: incomplete coverage (fixed quadratic map)" begin
    df = _load_simdata1_aj()
    ef, mats = build_error_function(df, _FEQ_AJ, :id, :t, :absS;
        algorithm=:iv, complete_coverage=false, precision_mode=:proxy)
    @test mats.jac_func !== nothing
    Random.seed!(1)
    ζpoints = [ones(5), randn(5), 3 .* randn(5)]
    _test_jac_vs_forwarddiff(ef, mats.jac_func, ζpoints)

    # the map is exactly quadratic ⇒ the Jacobian is exactly affine in ζ:
    # J(x) + J(y) == J(x + y) + J(0)
    x, y = randn(5), randn(5)
    @test mats.jac_func(x) + mats.jac_func(y) ≈ mats.jac_func(x + y) + mats.jac_func(zeros(5)) atol = 1e-10
end

@testset "analytic Jacobian: complete coverage, time-varying ζS (Mweights term)" begin
    df, ζtrue = _tv_zetaS_df()
    fml = @formula(q + id & endog(p) ~ 0)
    ef, mats = build_error_function(df, fml, :id, :t, :S; algorithm=:iv, precision_mode=:proxy)
    @test mats.jac_func !== nothing

    # teeth: the period Mweights are genuinely time-varying at the test points
    Mw = OptimalGIV.period_mweights(ζtrue, mats.C, mats.S, mats.obs_index)
    @test maximum(Mw) / minimum(Mw) > 1.2

    Random.seed!(2)
    ζpoints = [ζtrue, ζtrue .+ 0.3 .* randn(5), abs.(randn(5)) .+ 0.2]
    _test_jac_vs_forwarddiff(ef, mats.jac_func, ζpoints)

    # teeth: dropping the Mweights product-rule term must fail — compare against the
    # incomplete-coverage Jacobian rescaled naively (levels term matters)
    ef0, mats0 = build_error_function(df, fml, :id, :t, :S; algorithm=:iv,
        complete_coverage=false, precision_mode=:proxy)
    J_full = mats.jac_func(ζtrue)
    J_nomw = mats0.jac_func(ζtrue)
    @test norm(J_full - J_nomw) / norm(J_full) > 1e-3

    # direct call (reusable interface) matches the closure from build_error_function
    J_direct = mean_moment_jacobian(ζtrue, mats.uq, mats.uCp, mats.C, mats.S,
        mats.obs_index, true, mats.precision)
    @test J_direct == J_full
end

@testset "analytic Jacobian: excluded pairs" begin
    excl = Dict(1 => [2, 3], 4 => [5])
    Random.seed!(3)
    # incomplete coverage
    df1 = _load_simdata1_aj()
    ef1, mats1 = build_error_function(df1, _FEQ_AJ, :id, :t, :absS;
        algorithm=:iv, complete_coverage=false, precision_mode=:proxy, exclude_pairs=excl)
    @test any(mats1.obs_index.exclpairs)
    _test_jac_vs_forwarddiff(ef1, mats1.jac_func, [ones(5), randn(5)])
    # exclusion genuinely moves the Jacobian
    _, mats1_noexcl = build_error_function(df1, _FEQ_AJ, :id, :t, :absS;
        algorithm=:iv, complete_coverage=false, precision_mode=:proxy)
    @test norm(mats1.jac_func(ones(5)) - mats1_noexcl.jac_func(ones(5))) > 1e-8

    # complete coverage with time-varying ζS
    df2, ζtrue = _tv_zetaS_df()
    fml = @formula(q + id & endog(p) ~ 0)
    ef2, mats2 = build_error_function(df2, fml, :id, :t, :S;
        algorithm=:iv, precision_mode=:proxy, exclude_pairs=excl)
    @test any(mats2.obs_index.exclpairs)
    _test_jac_vs_forwarddiff(ef2, mats2.jac_func, [ζtrue, ζtrue .+ 0.3 .* randn(5)])
end

# fixed-precision moments with a CONSTANT loadings offset (λᵢ'λⱼ deducted from each
# pair), mirroring moment_conditions(:iv) with loadings held fixed at Λ.
function _moments_const_loadings(ζ, Λ, mats)
    obs_index, C, S, prec = mats.obs_index, mats.C, mats.S, mats.precision
    Nm = length(ζ)
    T = obs_index.T
    u = mats.uq .+ mats.uCp * ζ
    err = zeros(eltype(u), Nm, T)
    weightsum = zeros(eltype(u), Nm, T)
    OptimalGIV.fast_pass!(weightsum, err, u, C, S, prec, obs_index, Λ, size(Λ, 2))
    OptimalGIV.deduct_excluded_pairs!(err, weightsum, C, S, u, prec, obs_index, Λ, size(Λ, 2))
    err .*= OptimalGIV.period_mweights(ζ, C, S, obs_index)'
    momweight = sum(abs.(weightsum); dims=2)
    momweight ./= sum(momweight)
    err ./= momweight
    return vec(sum(err; dims=2) ./ T)
end

@testset "analytic Jacobian: constant loadings offset (nested-PC inner solve)" begin
    # externally supplied constant loadings enter the moments as a fixed expected_cov
    # deduction; the Jacobian picks it up only through the Mweights product-rule term.
    df, ζtrue = _tv_zetaS_df()
    fml = @formula(q + id & endog(p) ~ 0)
    _, mats = build_error_function(df, fml, :id, :t, :S; algorithm=:iv, precision_mode=:proxy)
    Random.seed!(4)
    Λ = 0.3 .* randn(mats.obs_index.N, 2)
    # reference map: the fixed-precision moments with the loadings held CONSTANT at Λ
    ef_const = x -> _moments_const_loadings(x, Λ, mats)
    for ζ in (ζtrue, ζtrue .+ 0.3 .* randn(5))
        J_fd = ForwardDiff.jacobian(ef_const, ζ)
        J_an = mean_moment_jacobian(ζ, mats.uq, mats.uCp, mats.C, mats.S,
            mats.obs_index, true, mats.precision; loadings_matrix=Λ)
        @test maximum(abs, J_an - J_fd) < 1e-8 * max(1.0, maximum(abs, J_fd))
    end
end

@testset "analytic Jacobian: roots identical to the FD/autodiff path" begin
    df = _load_simdata1_aj()
    for (alg, mode) in ((:iv, :proxy), (:iv_twopass, :proxy), (:iv, :twostep))
        m_an = giv(df, _FEQ_AJ, :id, :t, :absS; guess=ones(5), quiet=true, tol=1e-10,
            algorithm=alg, precision_mode=mode, jacobian=:analytic)
        m_fd = giv(df, _FEQ_AJ, :id, :t, :absS; guess=ones(5), quiet=true, tol=1e-10,
            algorithm=alg, precision_mode=mode, jacobian=:autodiff)
        @test m_an.converged && m_fd.converged
        @test maximum(abs, endog_coef(m_an) - endog_coef(m_fd)) < 1e-8
        @test vcov(m_an) ≈ vcov(m_fd) rtol = 1e-6
    end
    # :cue has no analytic Jacobian — `jacobian = :analytic` falls back to the
    # autodiff path, so results are byte-identical (no behavior change in :cue)
    m_cue_an = giv(df, _FEQ_AJ, :id, :t, :absS; guess=ones(5), quiet=true,
        algorithm=:iv, precision_mode=:cue, jacobian=:analytic)
    m_cue_fd = giv(df, _FEQ_AJ, :id, :t, :absS; guess=ones(5), quiet=true,
        algorithm=:iv, precision_mode=:cue, jacobian=:autodiff)
    @test endog_coef(m_cue_an) == endog_coef(m_cue_fd)
    @test vcov(m_cue_an) == vcov(m_cue_fd)
    # invalid option errors
    @test_throws ArgumentError giv(df, _FEQ_AJ, :id, :t, :absS; guess=ones(5), quiet=true,
        algorithm=:iv, precision_mode=:proxy, jacobian=:nonsense)
end

@testset "analytic Jacobian: fewer kernel evaluations than finite differences" begin
    df = _load_simdata1_aj()
    ef, mats = build_error_function(df, _FEQ_AJ, :id, :t, :absS; algorithm=:iv, precision_mode=:proxy)
    guess = ones(5)

    cnt_fd = Ref(0)
    res_fd = nlsolve(x -> (cnt_fd[] += 1; ef(x)), guess;
        method=:trust_region, autodiff=:central, ftol=1e-8)
    cnt_an = Ref(0)
    res_an = nlsolve(x -> (cnt_an[] += 1; ef(x)), mats.jac_func, guess;
        method=:trust_region, ftol=1e-8)

    @test converged(res_fd) && converged(res_an)
    @test maximum(abs, res_fd.zero - res_an.zero) < 1e-6
    @test cnt_an[] < cnt_fd[]
    @info "analytic Jacobian: kernel evaluations per solve (K = 5 moments)" fd_central = cnt_fd[] analytic = cnt_an[]
end
