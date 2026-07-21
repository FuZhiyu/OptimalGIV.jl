using Test, OptimalGIV, Random, LinearAlgebra
using OptimalGIV: build_error_function, mean_moment_conditions, mean_moment_jacobian
using OptimalGIV.NLsolve: nlsolve, converged
using ForwardDiff
using DataFrames, CSV, CategoricalArrays

# ---------------------------------------------------------------------------
# Analytic Jacobian of the fixed-precision moment kernel
# (giv-solver-stability/analytic-jacobian)
#
# With fixed precisions the residuals u = uq + uCp ζ are linear in ζ, so the
# Jacobian of the :iv/:iv_twopass moment map is exact in closed form (same O(N)
# pair-sum identity as the moments; constant momweight; optional frozen period
# scaling under complete coverage). These tests pin the hand-coded
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
# p_t clears the market exactly); reused from
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
        algorithm=:iv, complete_coverage=false, precision_weights=:raw_onestep)
    @test mats.jac_func !== nothing
    Random.seed!(1)
    ζpoints = [ones(5), randn(5), 3 .* randn(5)]
    _test_jac_vs_forwarddiff(ef, mats.jac_func, ζpoints)

    # the map is exactly quadratic ⇒ the Jacobian is exactly affine in ζ:
    # J(x) + J(y) == J(x + y) + J(0)
    x, y = randn(5), randn(5)
    @test mats.jac_func(x) + mats.jac_func(y) ≈ mats.jac_func(x + y) + mats.jac_func(zeros(5)) atol = 1e-10
end

@testset "analytic Jacobian: frozen complete-coverage period scaling" begin
    df, ζtrue = _tv_zetaS_df()
    fml = @formula(q + id & endog(p) ~ 0)
    ef_equal, mats = build_error_function(df, fml, :id, :t, :S;
        algorithm=:iv, complete_coverage=true, precision_weights=:raw_onestep)
    @test mats.jac_func !== nothing

    # teeth: the period Mweights are genuinely time-varying at the test points
    Mw = OptimalGIV.period_mweights(ζtrue, mats.C, mats.S, mats.obs_index)
    @test maximum(Mw) / minimum(Mw) > 1.2

    ef_frozen = x -> mean_moment_conditions(x, mats.uq, mats.uCp, mats.C, mats.S,
        mats.obs_index, true, Val(:iv); precision=mats.precision, Mweights=Mw)
    jac_frozen = x -> mean_moment_jacobian(x, mats.uq, mats.uCp, mats.C, mats.S,
        mats.obs_index, true, mats.precision; Mweights=Mw)

    Random.seed!(2)
    ζpoints = [ζtrue, ζtrue .+ 0.3 .* randn(5), abs.(randn(5)) .+ 0.2]
    _test_jac_vs_forwarddiff(ef_frozen, jac_frozen, ζpoints)
    # Frozen weights preserve the exact quadratic map and affine Jacobian.
    x, y = randn(5), randn(5)
    @test jac_frozen(x) + jac_frozen(y) ≈ jac_frozen(x + y) + jac_frozen(zeros(5)) atol = 1e-10

    # Teeth: frozen period weights remain in both the moments and Jacobian.
    @test norm(ef_frozen(ζtrue) - ef_equal(ζtrue)) / norm(ef_frozen(ζtrue)) > 1e-3
    @test norm(jac_frozen(ζtrue) - mats.jac_func(ζtrue)) / norm(jac_frozen(ζtrue)) > 1e-3

    # direct call (reusable interface) matches the closure from build_error_function
    J_direct = mean_moment_jacobian(ζtrue, mats.uq, mats.uCp, mats.C, mats.S,
        mats.obs_index, true, mats.precision; Mweights=Mw)
    @test J_direct == jac_frozen(ζtrue)
end

@testset "analytic Jacobian: excluded pairs" begin
    excl = Dict(1 => [2, 3], 4 => [5])
    Random.seed!(3)
    # incomplete coverage
    df1 = _load_simdata1_aj()
    ef1, mats1 = build_error_function(df1, _FEQ_AJ, :id, :t, :absS;
        algorithm=:iv, complete_coverage=false, precision_weights=:raw_onestep, exclude_pairs=excl)
    @test any(mats1.obs_index.exclpairs)
    _test_jac_vs_forwarddiff(ef1, mats1.jac_func, [ones(5), randn(5)])
    # exclusion genuinely moves the Jacobian
    _, mats1_noexcl = build_error_function(df1, _FEQ_AJ, :id, :t, :absS;
        algorithm=:iv, complete_coverage=false, precision_weights=:raw_onestep)
    @test norm(mats1.jac_func(ones(5)) - mats1_noexcl.jac_func(ones(5))) > 1e-8

    # complete coverage with time-varying ζS
    df2, ζtrue = _tv_zetaS_df()
    fml = @formula(q + id & endog(p) ~ 0)
    ef2, mats2 = build_error_function(df2, fml, :id, :t, :S;
        algorithm=:iv, complete_coverage=true, precision_weights=:raw_onestep, exclude_pairs=excl)
    @test any(mats2.obs_index.exclpairs)
    _test_jac_vs_forwarddiff(ef2, mats2.jac_func, [ζtrue, ζtrue .+ 0.3 .* randn(5)])
end

@testset "analytic Jacobian: automatic selection preserves the finite-difference root" begin
    df = _load_simdata1_aj()
    for alg in (:iv, :iv_twopass)
        ef, _ = build_error_function(df, _FEQ_AJ, :id, :t, :absS;
            algorithm=alg, complete_coverage=true, precision_weights=:raw_onestep)
        fd = nlsolve(ef, ones(5); method=:trust_region, autodiff=:central, ftol=1e-10)
        m = giv(df, _FEQ_AJ, :id, :t, :absS; guess=ones(5), quiet=true,
            algorithm=alg, precision_weights=:raw_onestep,
            complete_coverage=true,
            solver_options=(; method=:trust_region, autodiff=:central, ftol=1e-10,
                show_trace=false, iterations=100))
        @test m.converged && converged(fd)
        @test maximum(abs, endog_coef(m) - fd.zero) < 1e-8
    end

    # Removed top-level solver selectors have no compatibility aliases.
    @test_throws MethodError giv(df, _FEQ_AJ, :id, :t, :absS; quiet=true, complete_coverage=true,
        jacobian=:analytic)
    @test_throws MethodError giv(df, _FEQ_AJ, :id, :t, :absS; quiet=true, complete_coverage=true,
        method=:trust_region)
    @test_throws MethodError giv(df, _FEQ_AJ, :id, :t, :absS; quiet=true, complete_coverage=true,
        autodiff=:central)
end

@testset "analytic Jacobian: fewer kernel evaluations than finite differences" begin
    df = _load_simdata1_aj()
    ef, mats = build_error_function(df, _FEQ_AJ, :id, :t, :absS;
        algorithm=:iv, complete_coverage=true, precision_weights=:raw_onestep)
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
