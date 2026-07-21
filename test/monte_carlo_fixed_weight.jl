# ---------------------------------------------------------------------------
# Monte Carlo: efficiency and stability of fixed proxy precision weights
# (precision_mode = :proxy / :fixed two-step) vs continuously-updated CUE weights.
#
# Backs the researcher's conjecture that var(uqᵢ) is "close enough" to var(uᵢ):
# reports bias, RMSE, mean SE, and 95% coverage for the size-weighted aggregate
# elasticity (efficiency, measured from a good guess so all methods converge),
# plus convergence rate from generic initial guesses (ones / OLS) — the stability
# gain the fixed-weight mode is meant to deliver.
#
# Run:  julia --project=. test/monte_carlo_fixed_weight.jl
# Sections selectable via GIV_MC_SECTIONS (comma list of base,cue_favorable; default both).
# ---------------------------------------------------------------------------
using OptimalGIV, DataFrames, Random, LinearAlgebra, Statistics, Printf
using OptimalGIV: simulate_data, SimModel

# Scenarios drawn from test/simulate_data.jl SIMULATION_SCENARIOS, kept moderate.
const MC_SCENARIOS = [
    (label="baseline (N=10,σζ=1)",       params=(N=10, T=100, K=2, ushare=0.5, σζ=1.0, missingperc=0.0)),
    (label="concentrated (N=10,h=0.5)",  params=(N=10, T=100, K=2, ushare=0.5, σζ=1.0, h=0.5, missingperc=0.0)),
    (label="wider (N=20,T=200,σζ=1)",    params=(N=20, T=200, K=2, ushare=0.4, σζ=1.0, missingperc=0.0)),
]
const FORMULA = @formula(q + id & endog(p) ~ fe(id) & (η1 + η2) + 0)
const NREP = 150
# efficiency: good guess, tight tol.  stability: generic guess, capped iterations.
const SOLVER_EFF = (; ftol=1e-8, iterations=200)
const SOLVER_STAB = (; ftol=1e-6, iterations=100)

agg_truth(df) = (iddf = sort(unique(df, :id), :id); (sum(iddf.S .* iddf.ζ), iddf.S))

"""One estimation → (converged, agg_bias, covered, se_agg)."""
function one_fit(df, S, trueagg; guess, solver, formula=FORMULA, kw...)
    m = try
        giv(df, formula, :id, :t, :S; guess=guess, quiet=true, algorithm=:iv,
            solver_options=solver, kw...)
    catch
        return (false, missing, missing, missing)
    end
    m.converged || return (false, missing, missing, missing)
    estζ = endog_coef(m); Σζ = endog_vcov(m)
    estagg = sum(S .* estζ); seagg = sqrt(max(S' * Σζ * S, 0.0))
    bias = estagg - trueagg
    return (true, bias, abs(bias) <= 1.96 * seagg, seagg)
end

"""Two-step :fixed: proxy first pass, then precisions from first-step residual variance."""
function twostep_fit(df, S, trueagg; guess, solver)
    m1 = try
        giv(df, FORMULA, :id, :t, :S; guess=guess, quiet=true, algorithm=:iv,
            solver_options=solver, precision_mode=:proxy)
    catch
        return (false, missing, missing, missing)
    end
    m1.converged || return (false, missing, missing, missing)
    return one_fit(df, S, trueagg; guess=endog_coef(m1), solver=solver,
        precision_mode=:fixed, precision_weights=1 ./ m1.residual_variance)
end

function summarize(results)
    conv = [r[1] for r in results]; convrate = mean(conv)
    ok = findall(conv)
    isempty(ok) && return (convrate, NaN, NaN, NaN, NaN)
    bias = [results[i][2] for i in ok]; cov = [results[i][3] for i in ok]; se = [results[i][4] for i in ok]
    return (convrate, mean(bias), sqrt(mean(abs2, bias)), mean(se), mean(cov))
end

fit_all(dfs, N, guessfn, solver, fitter) = [begin
    df = dfs[i]; df.id = string.(df.id); trueagg, S = agg_truth(df)
    fitter(df, S, trueagg; guess=guessfn(df, N), solver=solver)
end for i in 1:length(dfs)]

trueguess(df, N) = sort(unique(df, :id), :id).ζ
onesguess(df, N) = ones(N)
olsguess(df, N) = nothing

function run_mc()
    println("Monte Carlo: fixed-weight vs CUE, $(NREP) reps/scenario\n"); flush(stdout)
    for scen in MC_SCENARIOS
        N = scen.params.N
        dfs = simulate_data(scen.params; Nsims=NREP, seed=42)
        methods = [
            ("CUE",      (df, S, ta; kw...) -> one_fit(df, S, ta; kw...)),
            ("proxy",    (df, S, ta; kw...) -> one_fit(df, S, ta; precision_mode=:proxy, kw...)),
            ("two-step", (df, S, ta; kw...) -> twostep_fit(df, S, ta; kw...)),
        ]
        println("=== $(scen.label) ===")
        @printf("%-9s %9s %8s %8s %7s | %9s %9s %9s\n",
            "method", "bias", "rmse", "meanSE", "cover", "conv:true", "conv:ones", "conv:OLS")
        for (name, fitter) in methods
            eff  = summarize(fit_all(dfs, N, trueguess, SOLVER_EFF, fitter))
            cones = summarize(fit_all(dfs, N, onesguess, SOLVER_STAB, fitter))[1]
            cols = summarize(fit_all(dfs, N, olsguess, SOLVER_STAB, fitter))[1]
            (_, bias, rmse, mse, cov) = eff
            @printf("%-9s %9.4f %8.4f %8.4f %7.3f | %9.3f %9.3f %9.3f\n",
                name, bias, rmse, mse, cov, eff[1], cones, cols)
        end
        println(); flush(stdout)
    end
end

# ---------------------------------------------------------------------------
# CUE-favorable scenario with oracle-weight limit
#
# Worst-case efficiency cost of fixed proxy weights in a DGP deliberately built
# to favor CUE, plus the oracle limit (true precisions 1/σᵤᵢ²).
#
# PRE-REGISTERED parameterization — chosen against the contamination ratio
# ζᵢ²·var(p̃)/σᵤᵢ² ONLY (target: elasticity term comparable to idio variance for
# the elastic entities), NEVER against estimator output:
#   N=10, T=100, K=2
#   h=0.5        concentrated sizes
#   σᵤcurv=0.5   real σᵤ heterogeneity (large entities much less volatile)
#   M=2, σζ=1.5  small mean elasticity (ζ̄ = 1/M = 0.5) with large spread
#   ushare=0.8   idiosyncratic shocks dominate the residualized price
#   ζalign=-1    largest ζ assigned to smallest-σᵤ entities (proxy misweights hardest)
#
# Estimated specification: 5 category elasticities, each category pooling two
# entities at opposite size extremes ((1,10),(2,9),...). With per-entity
# elasticities the system is one-moment-per-entity exactly identified and every
# precision weighting shares the same roots (verified: CUE ≡ proxy ≡ oracle to
# solver tolerance), so weighting can only matter when moments pool entities —
# exactly the Treasury setting (9–13 category elasticities over 57 sectors).
# Extreme-size pairing maximizes within-moment precision heterogeneity, the
# channel proxy weights can get wrong. True category ζ = S-weighted mean of the
# anti-aligned entity draws within the pair, recentred so S'ζ = 1/M; q is rebuilt
# from the simulated primitives (q = u + Λη − ζ_g·p), which keeps the DGP
# internally consistent since p = M·S'(u + Λη).
#
# Realized contamination at the frozen params under the category ζ (400 reps,
# computed from DGP primitives with η controlled, p̃ = M·S'u so
# var(p̃) = M²·ΣSⱼ²σᵤⱼ²; no estimation involved): per-rep max mean 2.32
# [q10 1.09, q90 3.81], per-rep median mean 0.37, share of entity-reps above
# 1.0 = 0.17 — elasticity term comparable to (or exceeding) idio variance for
# the elastic entities; target met, parameters frozen before the estimator arms.
# ---------------------------------------------------------------------------
const CUEFAV_PARAMS = (N=10, T=100, K=2, h=0.5, σᵤcurv=0.5, M=2.0, σζ=1.5, ushare=0.8, ζalign=-1)
const CUEFAV_NREP = 400
const CUEFAV_PAIR = Dict(1 => 1, 10 => 1, 2 => 2, 9 => 2, 3 => 3, 8 => 3, 4 => 4, 7 => 4, 5 => 5, 6 => 5)
const CUEFAV_FORMULA = @formula(q + grp & endog(p) ~ fe(id) & (η1 + η2) + 0)

"""Simulate reps keeping the DGP primitives needed for the oracle arm and the
contamination report. Entity order in estimation is sorted-string id order
("1","10","2",...,"9"), so the per-entity oracle precisions are permuted
accordingly; category coefficients/weights are in sorted group order g1..g5."""
function simulate_cuefav(params, nrep, seed)
    Random.seed!(seed)
    N = params.N
    perm = sortperm(string.(1:N))
    map(1:nrep) do _
        m = SimModel(; params...)
        df = DataFrame(m.data)
        su = copy(m.param.σᵤvec)            # final (rescaled) true σᵤ, numeric entity order
        S = m.param.constS
        ζdraw = collect(skipmissing(m.data.ζ))
        masks = [[CUEFAV_PAIR[i] == g for i in 1:N] for g in 1:5]
        ζg = [sum(S[mk] .* ζdraw[mk]) / sum(S[mk]) for mk in masks]
        ζent = [ζg[CUEFAV_PAIR[i]] for i in 1:N]
        shift = (1 / params.M - sum(S .* ζent)) / sum(S)
        ζg .+= shift
        ζent .+= shift
        idnum = parse.(Int, df.id)
        df.grp = string.("g", getindex.(Ref(CUEFAV_PAIR), idnum))
        df.ζ = ζent[idnum]
        df.q = df.u .+ df.λ1 .* df.η1 .+ df.λ2 .* df.η2 .- df.ζ .* df.p
        varp = params.M^2 * sum(S .^ 2 .* su .^ 2)
        Sg = [sum(S[mk]) for mk in masks]
        (; df, Sg, trueagg=sum(Sg .* ζg), ζg,
            oracle_w=1 ./ su[perm] .^ 2, contamination=ζent .^ 2 .* varp ./ su .^ 2)
    end
end

function cuefav_summarize(results)
    conv = [r[1] for r in results]
    ok = findall(conv)
    err = [results[i][2] for i in ok]
    cov = [results[i][3] for i in ok]
    return (convrate=mean(conv), bias=mean(err), sd=std(err), cover=mean(cov))
end

function run_cue_favorable_mc()
    println("=== CUE-favorable scenario with oracle limit, $(CUEFAV_NREP) reps ===")
    reps = simulate_cuefav(CUEFAV_PARAMS, CUEFAV_NREP, 42)
    contam = reduce(vcat, r.contamination for r in reps)
    permax = [maximum(r.contamination) for r in reps]
    permed = [median(r.contamination) for r in reps]
    @printf("contamination ζᵢ²var(p̃)/σᵤᵢ²: per-rep max mean %.2f [q10 %.2f, q90 %.2f], per-rep median mean %.2f, share>1 %.2f\n\n",
        mean(permax), quantile(permax, 0.1), quantile(permax, 0.9), mean(permed), mean(contam .> 1))
    arms = [
        ("CUE/true",    r -> one_fit(r.df, r.Sg, r.trueagg; guess=r.ζg, solver=SOLVER_EFF, formula=CUEFAV_FORMULA)),
        ("proxy/true",  r -> one_fit(r.df, r.Sg, r.trueagg; guess=r.ζg, solver=SOLVER_EFF, formula=CUEFAV_FORMULA, precision_mode=:proxy)),
        ("proxy/ones",  r -> one_fit(r.df, r.Sg, r.trueagg; guess=ones(5), solver=SOLVER_EFF, formula=CUEFAV_FORMULA, precision_mode=:proxy)),
        ("oracle/true", r -> one_fit(r.df, r.Sg, r.trueagg; guess=r.ζg, solver=SOLVER_EFF, formula=CUEFAV_FORMULA,
            precision_mode=:fixed, precision_weights=r.oracle_w)),
    ]
    @printf("%-11s %6s %9s %9s %7s\n", "arm", "conv", "bias", "empSD", "cover")
    sds = Dict{String,Float64}()
    for (name, fitter) in arms
        s = cuefav_summarize([fitter(r) for r in reps])
        sds[name] = s.sd
        @printf("%-11s %6.3f %9.4f %9.4f %7.3f\n", name, s.convrate, s.bias, s.sd, s.cover)
        flush(stdout)
    end
    @printf("\nheadline: SD_proxy/SD_oracle = %.3f   SD_proxy/SD_cue = %.3f\n\n",
        sds["proxy/true"] / sds["oracle/true"], sds["proxy/true"] / sds["CUE/true"])
end

const MC_SECTIONS = split(get(ENV, "GIV_MC_SECTIONS", "base,cue_favorable"), ",")
"base" in MC_SECTIONS && run_mc()
"cue_favorable" in MC_SECTIONS && run_cue_favorable_mc()
