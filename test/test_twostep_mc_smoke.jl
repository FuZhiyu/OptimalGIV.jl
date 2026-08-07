using Test, OptimalGIV, Random, LinearAlgebra, Statistics
using OptimalGIV: calculate_entity_variance, period_mweights, solve_vcov,
    build_error_function
using DataFrames, CategoricalArrays

# ---------------------------------------------------------------------------
# Cheap smoke tier of the complete-coverage two-step Monte Carlo
# (giv-solver-stability/suite-and-evidence-repair, umbrella decision 16).
#
# The full statistical acceptance bands live in the manual scripts
# test/monte_carlo_timevarying_mweights.jl and test/monte_carlo_fixed_weight.jl
# (run cadence documented in CLAUDE.md). Those bars used to be the ONLY guard on
# the two-step estimator, so the 2026-07-23 vcov-routing change rotted the
# recorded evidence for two days before anyone noticed. This tier runs a handful
# of SMALL seeded draws inside `Pkg.test` so a routing regression, an SE-formula
# transposition, or a gross SE-calibration break trips the suite immediately.
# It asserts deterministic structural invariants plus a WIDE SE-calibration
# sanity band (not the pre-registered scientific bands) — seeds are pinned, so
# the whole tier is reproducible, not flaky.
# ---------------------------------------------------------------------------

function _smoke_tv_draw(seed)
    Random.seed!(seed)
    N, T = 5, 60
    ζtrue = [0.5, 1.0, 1.5, 2.0, 3.0]
    Smat = rand(N, T) .^ 3 .+ 0.05
    Smat ./= sum(Smat; dims=1)
    umat = randn(N, T) .* (0.5 .+ rand(N))
    p = [dot(Smat[:, t], umat[:, t]) / dot(Smat[:, t], ζtrue) for t in 1:T]
    qmat = umat .- ζtrue * p'
    df = DataFrame(id=CategoricalArray(repeat(1:N, T)), t=repeat(1:T; inner=N),
        q=vec(qmat), p=repeat(p; inner=N), S=vec(Smat))
    return df, @formula(q + id & endog(p) ~ 0), ζtrue
end

@testset "two-step MC smoke: routing, finite SEs, frozen-sandwich equality" begin
    nrep = 20
    K = 5
    coefs = Matrix{Float64}(undef, K, nrep)
    ses = Matrix{Float64}(undef, K, nrep)
    for s in 1:nrep
        df, fml, ζtrue = _smoke_tv_draw(1000 + s)
        m1 = giv(df, fml, :id, :t, :S; guess=ζtrue, quiet=true, algorithm=:iv,
            complete_coverage=true, precision_weights=:raw_onestep)
        # default :auto route ⇒ information formula
        m2 = giv(df, fml, :id, :t, :S; guess=ζtrue, quiet=true, algorithm=:iv,
            complete_coverage=true, precision_weights=:twostep)
        # forced sandwich route on the same estimator
        m2s = giv(df, fml, :id, :t, :S; guess=ζtrue, quiet=true, algorithm=:iv,
            complete_coverage=true, precision_weights=:twostep, vcov=:sandwich)

        @test m1.converged && m2.converged && m2s.converged
        @test m2.vcov_method == :optimal
        @test m2s.vcov_method == :sandwich
        @test all(isfinite, vcov(m2)) && isposdef(Symmetric(vcov(m2)))
        @test all(isfinite, vcov(m2s)) && isposdef(Symmetric(vcov(m2s)))
        @test vcov(m2) != vcov(m2s)   # the two routes genuinely differ

        # anti-tautology tooth: the :sandwich route reproduces the hand-built frozen
        # step-1 weight bundle exactly; the :auto route does not.
        _, mats = build_error_function(df, fml, :id, :t, :S; algorithm=:iv,
            complete_coverage=true, precision_weights=:raw_onestep)
        u1 = mats.uq + mats.uCp * endog_coef(m1)
        w2 = 1 ./ calculate_entity_variance(u1, mats.obs_index)
        M1 = period_mweights(endog_coef(m1), mats.C, mats.S, mats.obs_index)
        u2 = mats.uq + mats.uCp * endog_coef(m2)
        _, Σfrozen = solve_vcov(u2, mats.S, mats.C, mats.uCp, mats.obs_index;
            precision=w2, Mweights=M1)
        @test vcov(m2s) == Σfrozen

        coefs[:, s] = endog_coef(m2)
        ses[:, s] = sqrt.(diag(vcov(m2)))
    end

    # Point estimates land near the truth (true-guess start on a well-behaved DGP).
    @test maximum(abs, vec(mean(coefs; dims=2)) - [0.5, 1.0, 1.5, 2.0, 3.0]) < 0.3

    # WIDE SE-calibration sanity: the information-formula SEs are the right order of
    # magnitude vs the across-rep empirical SD of the estimates. Deliberately loose
    # (the tight coverage/SE bands are the manual MC's job); only catches gross
    # breakage such as a missing 1/T or a transposed bread.
    empirical_sd = vec(std(coefs; dims=2))
    mean_se = vec(mean(ses; dims=2))
    ratio = median(mean_se ./ empirical_sd)
    @info "two-step MC smoke: median information-formula SE / empirical SD" ratio
    @test 0.5 <= ratio <= 2.0
end
