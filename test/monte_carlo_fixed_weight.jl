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
# ---------------------------------------------------------------------------
using OptimalGIV, DataFrames, Random, LinearAlgebra, Statistics, Printf
using OptimalGIV: simulate_data

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
function one_fit(df, S, trueagg; guess, solver, kw...)
    m = try
        giv(df, FORMULA, :id, :t, :S; guess=guess, quiet=true, algorithm=:iv,
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

run_mc()
