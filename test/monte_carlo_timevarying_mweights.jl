# ---------------------------------------------------------------------------
# Monte Carlo: complete-coverage two-step with time-varying period weights.
#
# Run from the package root:
#   julia --project=. test/monte_carlo_timevarying_mweights.jl
#
# PRE-REGISTERED before the first estimator run (2026-07-21):
#   seed=20260721, N=8, G=4, T=160, reps=250.
#   Group elasticities are (0.4, 0.8, 1.6, 3.2). Deterministic group shares
#   rotate sinusoidally with amplitude 0.75 and quarter-cycle phase offsets;
#   each group contains two entities with a fixed 55/45 split. Thus population
#   aggregate elasticity varies over time without changing the correctly
#   specified group coefficients. Entity shocks are independent Gaussian with
#   fixed heterogeneous standard deviations.
#
# Pre-registered acceptance bands (not selected from estimator output):
#   population M max/min >= 1.75 and coefficient of variation >= 0.20;
#   each arm convergence >= 0.95 and |aggregate bias| <= 0.10;
#   each arm 95% coverage in [0.88, 0.99];
#   two-step mean SE / empirical SD in [0.75, 1.25];
#   SD(two-step)/SD(CUE) <= 1.20 and SD(two-step)/SD(oracle) <= 1.25;
#   median relative L2 gap M(zetatilde) vs M(zetahat2) >= 1e-4.
# Any failed band remains visible as @test_broken after diagnosis; bands are not
# changed after the run.
# ---------------------------------------------------------------------------
using OptimalGIV, DataFrames, CSV, Random, LinearAlgebra, Statistics, Test
using OptimalGIV: calculate_entity_variance, period_mweights, solve_vcov

const TVM_SEED = 20260721
const TVM_N = 8
const TVM_G = 4
const TVM_T = 160
const TVM_NREP = 250
const TVM_ZETA = [0.4, 0.8, 1.6, 3.2]
const TVM_SIGMA_U = [0.5, 1.2, 0.7, 1.4, 0.6, 1.0, 0.8, 1.3]
const TVM_FORMULA = @formula(q + grp & endog(p) ~ 0)
const TVM_SOLVER = (; ftol=1e-7, iterations=150)

const TVM_BANDS = (
    min_population_M_ratio=1.75,
    min_population_M_cv=0.20,
    min_convergence=0.95,
    max_abs_bias=0.10,
    coverage=(0.88, 0.99),
    twostep_se_sd=(0.75, 1.25),
    max_twostep_cue_sd_ratio=1.20,
    max_twostep_oracle_sd_ratio=1.25,
    min_median_M_gap=1e-4,
)

function tvm_sizes()
    group_share = [
        (1 + 0.75 * sin(2pi * (t - 1) / TVM_T + (g - 1) * pi / 2)) / TVM_G
        for g in 1:TVM_G, t in 1:TVM_T
    ]
    S = zeros(TVM_N, TVM_T)
    for g in 1:TVM_G
        S[2g - 1, :] .= 0.55 .* group_share[g, :]
        S[2g, :] .= 0.45 .* group_share[g, :]
    end
    return S
end

const TVM_S = tvm_sizes()
const TVM_GROUP = repeat(1:TVM_G; inner=2)
const TVM_ZETA_ENTITY = TVM_ZETA[TVM_GROUP]
const TVM_AGG_LOAD = [mean(vec(sum(TVM_S[TVM_GROUP .== g, :]; dims=1))) for g in 1:TVM_G]

function tvm_replication()
    u = TVM_SIGMA_U .* randn(TVM_N, TVM_T)
    aggregate = vec(TVM_ZETA_ENTITY' * TVM_S)
    p = vec(sum(TVM_S .* u; dims=1)) ./ aggregate
    q = u .- TVM_ZETA_ENTITY * p'
    df = DataFrame(
        id=repeat(string.(1:TVM_N); outer=TVM_T),
        grp=repeat(string.("g", TVM_GROUP); outer=TVM_T),
        t=repeat(1:TVM_T; inner=TVM_N),
        S=vec(TVM_S),
        q=vec(q),
        p=repeat(p; inner=TVM_N),
    )
    return (; df, oracle_precision=1 ./ TVM_SIGMA_U .^ 2)
end

function aggregate_result(zetahat, Sigma)
    truth = dot(TVM_AGG_LOAD, TVM_ZETA)
    estimate = dot(TVM_AGG_LOAD, zetahat)
    se = sqrt(max(dot(TVM_AGG_LOAD, Sigma * TVM_AGG_LOAD), 0.0))
    error = estimate - truth
    return (; error, se, covered=abs(error) <= 1.96 * se)
end

function fixed_fit(mats, guess, precision, Mweights)
    fit = try
        estimate_giv(mats.uq, mats.uCp, mats.C, mats.S, mats.obs_index, Val(:iv);
            guess, quiet=true, complete_coverage=true, solver_options=TVM_SOLVER,
            precision, Mweights)
    catch
        return (; converged=false, result=nothing, zetahat=fill(NaN, TVM_G))
    end
    zetahat, converged = fit
    if !converged
        return (; converged=false, result=nothing, zetahat)
    end
    uhat = mats.uq + mats.uCp * zetahat
    _, Sigma = solve_vcov(uhat, mats.S, mats.C, mats.uCp, mats.obs_index;
        precision, Mweights)
    return (; converged=true, result=aggregate_result(zetahat, Sigma), zetahat)
end

function run_replication(r)
    df = copy(r.df)
    _, mats = build_error_function(df, TVM_FORMULA, :id, :t, :S;
        algorithm=:iv, complete_coverage=true, precision_weights=:raw_onestep)
    guess = ones(TVM_G)

    m1 = try
        giv(df, TVM_FORMULA, :id, :t, :S; guess, quiet=true,
            algorithm=:iv, complete_coverage=true, solver_options=TVM_SOLVER,
            precision_weights=:raw_onestep)
    catch
        nothing
    end
    m2 = try
        giv(df, TVM_FORMULA, :id, :t, :S; guess, quiet=true,
            algorithm=:iv, complete_coverage=true, solver_options=TVM_SOLVER,
            precision_weights=:twostep)
    catch
        nothing
    end
    mcue = try
        giv(df, TVM_FORMULA, :id, :t, :S; guess, quiet=true,
            algorithm=:iv, complete_coverage=true, solver_options=TVM_SOLVER,
            precision_weights=:cue)
    catch
        nothing
    end

    twostep = nothing
    Mgap = missing
    sandwich_match = false
    twostep_converged = !isnothing(m1) && !isnothing(m2) && m1.converged && m2.converged
    if twostep_converged
        u1 = mats.uq + mats.uCp * endog_coef(m1)
        precision2 = 1 ./ calculate_entity_variance(u1, mats.obs_index)
        M1 = period_mweights(endog_coef(m1), mats.C, mats.S, mats.obs_index)
        M2 = period_mweights(endog_coef(m2), mats.C, mats.S, mats.obs_index)
        u2 = mats.uq + mats.uCp * endog_coef(m2)
        _, Sigma_frozen = solve_vcov(u2, mats.S, mats.C, mats.uCp, mats.obs_index;
            precision=precision2, Mweights=M1)
        sandwich_match = endog_vcov(m2) == Sigma_frozen
        Mgap = norm(M1 - M2) / norm(M1)
        twostep = aggregate_result(endog_coef(m2), endog_vcov(m2))
    end

    cue_converged = !isnothing(mcue) && mcue.converged
    cue = cue_converged ? aggregate_result(endog_coef(mcue), endog_vcov(mcue)) : nothing
    Moracle = period_mweights(TVM_ZETA, mats.C, mats.S, mats.obs_index)
    oracle = fixed_fit(mats, guess, r.oracle_precision, Moracle)
    return (; twostep_converged,
        cue_converged, oracle_converged=oracle.converged,
        twostep, cue, oracle=oracle.result, Mgap, sandwich_match)
end

function summarize_arm(results, arm)
    converged = getproperty.(results, Symbol(arm, "_converged"))
    fits = [getproperty(r, arm) for (r, ok) in zip(results, converged) if ok]
    errors = getproperty.(fits, :error)
    ses = getproperty.(fits, :se)
    return (; arm=String(arm), convergence=mean(converged), bias=mean(errors),
        empirical_sd=std(errors), mean_se=mean(ses), median_se=median(ses),
        coverage=mean(getproperty.(fits, :covered)), n_success=length(fits))
end

function run_tvm_mc()
    @assert all(isapprox.(vec(sum(TVM_S; dims=1)), 1.0; atol=1e-14))
    Random.seed!(TVM_SEED)
    results = [run_replication(tvm_replication()) for _ in 1:TVM_NREP]
    summaries = [summarize_arm(results, arm) for arm in (:twostep, :cue, :oracle)]
    summary = DataFrame(summaries)

    population_aggregate = vec(TVM_ZETA_ENTITY' * TVM_S)
    population_M = (1 ./ population_aggregate) ./ sum(1 ./ population_aggregate)
    M_ratio = maximum(population_M) / minimum(population_M)
    M_cv = std(population_M) / mean(population_M)
    Mgaps = collect(skipmissing(getproperty.(results, :Mgap)))
    median_M_gap = median(Mgaps)
    sandwich_matches = [r.sandwich_match for r in results if r.twostep_converged]
    sandwich_match_rate = mean(sandwich_matches)

    summary.population_M_ratio .= M_ratio
    summary.population_M_cv .= M_cv
    summary.median_first_second_M_gap .= median_M_gap
    summary.twostep_frozen_sandwich_match_rate .= sandwich_match_rate
    summary.seed .= TVM_SEED
    summary.N .= TVM_N
    summary.T .= TVM_T
    summary.replications .= TVM_NREP

    outpath = joinpath(@__DIR__, "..", "simresults", "timevarying_mweights_performance.csv")
    CSV.write(outpath, summary)
    show(stdout, MIME("text/plain"), summary); println()

    byarm = Dict(row.arm => row for row in eachrow(summary))
    @test M_ratio >= TVM_BANDS.min_population_M_ratio
    @test M_cv >= TVM_BANDS.min_population_M_cv
    @test median_M_gap >= TVM_BANDS.min_median_M_gap
    @test sandwich_match_rate == 1.0
    for row in eachrow(summary)
        @test row.convergence >= TVM_BANDS.min_convergence
        @test abs(row.bias) <= TVM_BANDS.max_abs_bias
        @test TVM_BANDS.coverage[1] <= row.coverage <= TVM_BANDS.coverage[2]
    end
    twostep = byarm["twostep"]
    cue = byarm["cue"]
    oracle = byarm["oracle"]
    @test TVM_BANDS.twostep_se_sd[1] <= twostep.mean_se / twostep.empirical_sd <= TVM_BANDS.twostep_se_sd[2]
    @test twostep.empirical_sd / cue.empirical_sd <= TVM_BANDS.max_twostep_cue_sd_ratio
    @test twostep.empirical_sd / oracle.empirical_sd <= TVM_BANDS.max_twostep_oracle_sd_ratio
    return summary
end

run_tvm_mc()
