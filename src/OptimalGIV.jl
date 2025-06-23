module OptimalGIV
using Random
using Base.Threads
using DataFrames
using LinearAlgebra
using Optim
using NLsolve
using Parameters
using StatsModels
using StatsModels: apply_schema, schema, hasintercept, InterceptTerm, FullRank, collect_matrix_terms
using Roots
using Distributions
using Reexport
using StatsBase
using StatsFuns
using Tables
using PrecompileTools
using FixedEffectModels: parse_fe, parse_fixedeffect, fe, FixedEffectTerm, FixedEffectModel, AbstractFixedEffectSolver, invsym!, has_fe
using FixedEffects: solve_residuals!, solve_coefficients!
@reexport using StatsAPI

include("givmodels.jl")
include("interface.jl")
include("estimation.jl")
include("scalar_search.jl")
include("utils/formula.jl")
include("utils/ols_fe_solver.jl")
# include("utils/delta_method.jl")

include("simulation.jl")
# include("gmm.jl")

export GIVModel
export @formula, endog
export giv,
    estimate_giv, create_coef_dataframe, preprocess_dataframe, get_coefnames, build_error_function, simulate_data
export coef,
    coefnames,
    coeftable,
    responsename,
    vcov,
    stderror,
    nobs,
    dof,
    fe,
    dof_residual,
    islinear,
    confint

@setup_workload begin
    df = DataFrame(;
        id=string.([1, 2, 3, 1, 2, 3, 1, 2, 3]),
        t=[1, 1, 1, 2, 2, 2, 3, 3, 3],
        q=[1.0; -0.5; -2.0; -1.0; 1.0; -1.0; 2.0; 0.0; -2.0],
        p=[1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -2.0, -2.0, -2.0],
        S=[1.0, 2.0, 0.5, 1.0, 2.0, 0.5, 1.0, 2.0, 0.5,],
        η=[-1.0, -1.0, -1.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0],
    )
    f = @formula(q + id & endog(p) ~ id & η + fe(id))
    kp = (; quiet=true, save=:all)
    @compile_workload begin
        giv(df, f, :id, :t, :S; algorithm=:scalar_search, guess=Dict("Aggregate" => 1.0), kp...)
        giv(df, f, :id, :t, :S; algorithm=:debiased_ols, kp...)
        giv(df, f, :id, :t, :S; algorithm=:iv, kp...)
        giv(df, f, :id, :t, :S; algorithm=:iv_twopass, kp...)
    end
end

end