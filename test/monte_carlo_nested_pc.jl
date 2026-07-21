# ---------------------------------------------------------------------------
# Monte Carlo + performance benchmark: nested PC solver vs one-step, for
# internal-PC (`pc(k)`) specifications where the factor controls are EXCLUDED
# (genuinely latent), matching the configurations where pc(k) specs have shown
# bias, convergence failures, and multiple roots (the removed pc_two_root_analysis.jl).
#
# Three arms over the same replications:
#   (a) CUE   one-step   — status quo (precision_mode=:cue,   pc_solver=:onestep)
#   (b) proxy one-step   — fixed weights, current solve (precision_mode=:proxy, :onestep)
#   (c) proxy nested     — outer PCA / inner fixed-quadratic ζ solve w/ analytic Jac
#                          (precision_mode=:proxy, pc_solver=:nested)
# Arms (b) and (c) share the SAME fixed proxy weights and differ ONLY in the solve
# mechanism, so their comparison isolates the nested structure + analytic Jacobian.
#
# Statistical: convergence rate from multiple guesses (true / ones / OLS default),
# bias and empirical SD of ζ̂ and of the aggregate ζS, and a root-agreement /
# multiplicity check across guesses. (vcov is NaN for n_pcs > 0, so no coverage.)
# Performance: pure-solver wall-clock and PCA-call / kernel-eval counts (the
# PCA-in-the-loop cost the nested method is meant to remove), MC panel + a larger
# production-scale panel.
#
# Run:  julia --project=. test/monte_carlo_nested_pc.jl
# Sections via GIV_MC_SECTIONS (comma list of stat,perf; default both).
# Quick smoke: GIV_MC_QUICK=1 (few reps).
# ---------------------------------------------------------------------------
using OptimalGIV, DataFrames, Random, Statistics, LinearAlgebra, Printf, NLsolve
using OptimalGIV: simulate_data, build_error_function, mean_moment_conditions, nested_pc_solve
using OptimalGIV.HeteroPCA: DeflatedHeteroPCA

const QUICK = get(ENV, "GIV_MC_QUICK", "0") == "1"

# Realistic production PCA settings (the package default), so the one-step
# PCA-in-the-loop cost reflects real usage rather than an inflated maxiter.
const PCAOPT = (; impute_method=:zero, demean=false, maxiter=100,
    algorithm=DeflatedHeteroPCA(t_block=10), suppress_warnings=true, abstol=1e-8)

# ---- pre-registered DGP scenarios (factors latent: `pc(k)`, no η controls) ----
# Homogeneous single-elasticity two-root config, verbatim from the removed
# test/pc_two_root_analysis.jl (N=40, T=400, K=2, M=0.5 ⇒ true ζ=2, σζ=0). This is
# the fixture where pc(2) specs exhibited two roots. 1 coefficient ⇒ clean scalar
# root analysis.
const HOMOG = (N=40, T=400, K=2, M=0.5, σζ=0.0, σp=2.0, σᵤcurv=0.1, h=0.2, ushare=0.5, missingperc=0.0)
const HOMOG_F = @formula(q + endog(p) ~ 0 + pc(2))

# Heterogeneous variant (pre-registered, planner guidance): σζ = 1 so entity
# elasticities are heterogeneous, pooled into size-ordered category coefficients
# so weighting/PCA interactions bite over a multi-dim solve. Kept small in (N,T)
# because CUE one-step with a multi-coef pc(k) spec is very expensive
# (FD Jacobian = (2K+1) PCA-in-the-loop evaluations per iteration).
const HETER = (N=30, T=120, K=2, M=0.5, σζ=1.0, σp=2.0, σᵤcurv=0.1, h=0.2, ushare=0.5, missingperc=0.0)
const HETER_NGRP = 4
const HETER_F = @formula(q + grp & endog(p) ~ 0 + pc(2))

const NREP = QUICK ? 6 : 100
const NREP_HET = QUICK ? 6 : 40
const SOLVER = (; ftol=1e-6, iterations=150)

# Assign size-ordered category groups to a simulated df (largest→group 1, etc.).
function add_groups!(df, ngrp)
    iddf = sort(unique(df, :id), :S, rev=true)
    ids = iddf.id
    grp = Dict(ids[i] => string("g", cld(i * ngrp, length(ids))) for i in eachindex(ids))
    df.grp = getindex.(Ref(grp), df.id)
    return df
end

# True aggregate ζS = Σ Sᵢ ζᵢ (size weights are constant over t in the DGP).
function true_agg(df)
    iddf = sort(unique(df, :id), :id)
    sum(iddf.S .* iddf.ζ)
end

# ---------------------------------------------------------------------------
# Statistical MC
# ---------------------------------------------------------------------------
"""One giv fit → (converged, ζ̂vec, aggζ̂). On non-convergence returns the final
iterate (landing point) with converged=false; only a thrown error returns empty."""
function fit_arm(df, formula, guess; precision_mode, pc_solver)
    m = try
        giv(df, formula, :id, :t, :S; guess=guess, quiet=true, algorithm=:iv,
            precision_mode=precision_mode, pc_solver=pc_solver,
            pca_option=PCAOPT, solver_options=merge(SOLVER, (; show_trace=false)),
            return_vcov=false)
    catch
        return (false, Float64[], NaN)
    end
    agg = agg_coef(m)
    aggsc = agg isa AbstractVector ? mean(agg) : agg
    return (m.converged, collect(endog_coef(m)), aggsc)
end

# distinct-root count: cluster solutions within tol; report max pairwise gap.
function root_multiplicity(sols; tol=1e-2)
    isempty(sols) && return (0, NaN)
    reps = Vector{Vector{Float64}}()
    for s in sols
        any(norm(s .- r) <= tol for r in reps) || push!(reps, s)
    end
    maxgap = length(sols) < 2 ? 0.0 : maximum(norm(sols[i] .- sols[j]) for i in 1:length(sols) for j in i+1:length(sols))
    return (length(reps), maxgap)
end

# arms::Vector of (name, precision_mode, pc_solver). When `report_drift`, also
# reports the mean landing-point drift ‖ζ̂−ζ_true‖ from the true guess over ALL
# reps (converged or not), exposing pc-spec pathology that convergence-gated
# accuracy hides.
function run_stat(label, params, formula, nrep, ngrp, arms; report_drift=false)
    println("=== STATISTICAL MC: $label ($(nrep) reps) ===")
    dfs = simulate_data(params; Nsims=nrep, seed=42)
    trueζsc = 1 / params.M
    for (name, pm, ps) in arms
        aggs = Float64[]                 # agg bias over converged true-guess reps
        drifts = Float64[]               # ‖ζ̂−ζtrue‖ from true guess, ALL reps
        conv_true = 0; conv_ones = 0; conv_ols = 0
        distinct_roots = Int[]; maxgaps = Float64[]
        for i in 1:nrep
            df = copy(dfs[i]); df.id = string.(df.id)
            ngrp > 0 && add_groups!(df, ngrp)
            ta = true_agg(df)
            ncoef = ngrp > 0 ? ngrp : 1
            ζtrue = ngrp > 0 ? _grp_true(df, ngrp) : [trueζsc]
            r_true = fit_arm(df, formula, ζtrue; precision_mode=pm, pc_solver=ps)
            r_ones = fit_arm(df, formula, ones(ncoef); precision_mode=pm, pc_solver=ps)
            r_ols = fit_arm(df, formula, nothing; precision_mode=pm, pc_solver=ps)
            conv_true += r_true[1]; conv_ones += r_ones[1]; conv_ols += r_ols[1]
            r_true[1] && push!(aggs, r_true[3] - ta)
            isempty(r_true[2]) || push!(drifts, norm(r_true[2] .- ζtrue))
            sols = [r[2] for r in (r_true, r_ones, r_ols) if r[1]]
            if !isempty(sols)
                nr, gap = root_multiplicity(sols)
                push!(distinct_roots, nr); push!(maxgaps, gap)
            end
        end
        aggbias = isempty(aggs) ? NaN : mean(aggs)
        aggsd = isempty(aggs) ? NaN : std(aggs)
        multi = isempty(distinct_roots) ? NaN : mean(distinct_roots .> 1)
        gapmax = isempty(maxgaps) ? NaN : maximum(maxgaps)
        driftm = isempty(drifts) ? NaN : mean(drifts)
        base = @sprintf("%-13s conv true %.2f ones %.2f OLS %.2f | share>1root %.2f maxgap %.3g",
            name, conv_true/nrep, conv_ones/nrep, conv_ols/nrep, multi, gapmax)
        if report_drift
            println(base * @sprintf(" | drift‖ζ̂−ζ*‖ %.3f", driftm))
        else
            println(base * @sprintf(" | agg bias %+8.4f empSD %7.4f", aggbias, aggsd))
        end
        flush(stdout)
    end
    println()
end

# true per-group ζ (S-weighted mean of member entity ζ), sorted-group order g1..gk.
function _grp_true(df, ngrp)
    d = combine(groupby(unique(df, :id), :grp)) do sub
        (; ζ=sum(sub.S .* sub.ζ) / sum(sub.S))
    end
    sort!(d, :grp)
    return d.ζ
end

# ---------------------------------------------------------------------------
# Performance benchmark: pure-solver wall-clock + PCA/kernel-eval counts
# ---------------------------------------------------------------------------
# Counting wrapper: each call to the one-step moment fn runs one PCA extraction.
mutable struct Counter; n::Int; end

function onestep_solve_counted(elem, guess; precision)
    cnt = Counter(0)
    fmom = x -> (cnt.n += 1; mean_moment_conditions(x, elem.uq, elem.uCp, elem.C, elem.S,
        elem.obs_index, elem.complete_coverage, Val(:iv), elem.n_pcs, PCAOPT; precision=precision))
    res = nlsolve(fmom, guess; method=:trust_region, autodiff=:central, SOLVER...)
    return res.zero, res.f_converged, cnt.n
end

# median @elapsed over `reps` timed solves (each on its own rep), after a warmup.
function _timed(f, dfs, reps)
    f(1)  # warmup (compile)
    ts = Float64[]
    for i in 1:reps
        push!(ts, @elapsed f(i))
    end
    median(ts)
end

function bench_config(label, params, ngrp, formula, reps)
    ncoef = ngrp == 0 ? 1 : ngrp
    println("=== PERFORMANCE: $label (N=$(params.N), T=$(params.T), coefs=$(ncoef), median of $(reps) reps) ===")
    dfs = simulate_data(params; Nsims=reps + 1, seed=7)
    trueζsc = 1 / params.M
    # pre-build matrices per rep (shared cost, excluded from the solve timing).
    prep = map(1:reps+1) do i
        df = copy(dfs[i]); df.id = string.(df.id)
        ngrp > 0 && add_groups!(df, ngrp)
        g = ngrp > 0 ? _grp_true(df, ngrp) : [trueζsc]
        _, ep = build_error_function(df, formula, :id, :t, :S; algorithm=:iv, precision_mode=:proxy, pca_option=PCAOPT, quiet=true)
        _, ec = build_error_function(df, formula, :id, :t, :S; algorithm=:iv, precision_mode=:cue, pca_option=PCAOPT, quiet=true)
        # DGP satisfies market clearing (S'q = 0) ⇒ complete coverage, as giv auto-detects.
        (; ep=merge(ep, (; complete_coverage=true)), ec=merge(ec, (; complete_coverage=true)), g)
    end
    cue_f(i) = onestep_solve_counted(prep[i].ec, copy(prep[i].g); precision=nothing)
    prx_f(i) = onestep_solve_counted(prep[i].ep, copy(prep[i].g); precision=prep[i].ep.precision)
    nes_f(i) = nested_pc_solve(copy(prep[i].g), prep[i].ep.uq, prep[i].ep.uCp, prep[i].ep.C,
        prep[i].ep.S, prep[i].ep.obs_index, true, Val(:iv); precision=prep[i].ep.precision,
        n_pcs=prep[i].ep.n_pcs, pca_option=PCAOPT, solver_options=SOLVER)

    t_cue = _timed(cue_f, prep, reps)
    t_prx = _timed(prx_f, prep, reps)
    t_nes = _timed(nes_f, prep, reps)

    # deterministic counts from a representative solve (rep 1).
    _, _, n_cue = cue_f(1)
    _, _, n_prx = prx_f(1)
    _, _, tr = nes_f(1)
    pca_nest = tr.outer_iters + 1

    @printf("%-13s %11s %12s %-24s\n", "arm", "median ms", "PCA calls", "kernel work")
    @printf("%-13s %11.2f %12d %-24s\n", "CUE 1-step", 1e3 * t_cue, n_cue, "$(n_cue) kernel evals")
    @printf("%-13s %11.2f %12d %-24s\n", "proxy 1-step", 1e3 * t_prx, n_prx, "$(n_prx) kernel evals")
    @printf("%-13s %11.2f %12d %-24s\n", "proxy nested", 1e3 * t_nes, pca_nest,
        "$(tr.outer_iters) outer / $(tr.total_inner) inner")
    @printf("speedup nested vs proxy-1step = %.2fx ; vs CUE-1step = %.2fx ; PCA-call reduction vs proxy-1step = %.1fx\n\n",
        t_prx / t_nes, t_cue / t_nes, n_prx / pca_nest)
    flush(stdout)
end

# ---------------------------------------------------------------------------
const SECTIONS = split(get(ENV, "GIV_MC_SECTIONS", "stat,perf"), ",")

if "stat" in SECTIONS
    # Homogeneous two-root config: all three arms converge, so report agg accuracy
    # and multi-root behavior across guesses.
    run_stat("homogeneous two-root (1 coef)", HOMOG, HOMOG_F, NREP, 0,
        [("CUE  1-step", :cue, :onestep), ("proxy 1-step", :proxy, :onestep), ("proxy nested", :proxy, :nested)])
    # Heterogeneous grouped config: pc-spec pathology (bias/non-convergence) — report
    # landing-point drift. CUE excluded (multi-coef CUE-pc is prohibitively slow and
    # known-pathological); the adoption-relevant comparison is proxy 1-step vs nested.
    run_stat("heterogeneous grouped ($(HETER_NGRP) coef)", HETER, HETER_F, NREP_HET, HETER_NGRP,
        [("proxy 1-step", :proxy, :onestep), ("proxy nested", :proxy, :nested)]; report_drift=true)
end

if "perf" in SECTIONS
    NT = QUICK ? 2 : 4
    bench_config("MC panel, homogeneous 1 coef", HOMOG, 0, HOMOG_F, NT)
    bench_config("MC panel, grouped 6 coef", (N=42, T=250, K=2, M=0.5, σζ=1.0, σp=2.0, σᵤcurv=0.1, h=0.2, ushare=0.5, missingperc=0.0), 6, HETER_F, NT)
    bench_config("larger panel, grouped 10 coef", (N=100, T=150, K=2, M=0.5, σζ=1.0, σp=2.0, σᵤcurv=0.1, h=0.2, ushare=0.5, missingperc=0.0), 10, HETER_F, NT)
end
