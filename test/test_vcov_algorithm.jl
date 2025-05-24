using OptimalGIV
using Einsum, LinearAlgebra, Test
using StatsBase, Random
Random.seed!(2)
N, T, Nmom = 5, 10, 7
qmat = rand(N, T)
Cts = zeros(N, T, Nmom)
[Cts[i, :, i] .= 1 for i in 1:N]
Cts[1, 6:10, 6] .= 1.0
Cts[:, :, 7] = rand(N, T)
p = rand(T)
Cpts = Cts .* p'
Smat = rand(N) * ones(1, T)

##================== test construction of covariance tensors ==================##
Cqq, CqCp, CCpq, CCpCp, qq, Cpq, CpCp = GIV.compuate_covariance_tensors(qmat, Cpts, Cts)

(@einsum Cqq2[i, g, j] := Cts[i, t, g] * qmat[j, t] * qmat[i, t]) ./= T-1
@test norm(Cqq2 - Array(Cqq)) < 1e-12

(@einsum CqCp2[i, g, k, j] := Cts[i, t, g] * qmat[i, t] * Cpts[j, t, k]) ./= T-1
@test norm(CqCp2 - Array(CqCp)) < 1e-12

(@einsum CCpq2[i, g, k, j] := Cts[i, t, g] * Cpts[i, t, k] * qmat[j, t]) ./= T-1
@test norm(CCpq2 - Array(CCpq)) < 1e-12

(@einsum CCpCp2[i, g, k, j, l] := Cts[i, t, g] * Cpts[i, t, k] * Cpts[j, t, l]) ./= T-1
@test norm(CCpCp2 - Array(CCpCp)) < 1e-12

(@einsum Cpq2[i, g] := Cpts[i, t, g] * qmat[i, t]) ./= T-1
@test norm(Cpq2 - Array(Cpq)) < 1e-12

(@einsum CpCp2[i, g, k] := Cpts[i, t, g] * Cpts[i, t, k]) ./= T-1
@test norm(CpCp2 - Array(CpCp)) < 1e-12

##================== test the moment condition ==================##
ζ = rand(Nmom)
σu²vec = GIV.compute_u_variance(ζ, qq, Cpq, CpCp)
u = qmat + dropdims(sum(Cpts .* reshape(ζ, 1, 1, Nmom), dims = 3), dims = 3)
σu²vec2 = vec(sum(u .^ 2, dims = 2)/(T-1))
@test norm(σu²vec2 - σu²vec) < 1e-12


##================== test the err_ig ==================##
err_ig = GIV.compute_err_ig(ζ, Cqq, CqCp, CCpq, CCpCp, zeros(Bool, N, N))
(@einsum err_ig2[i,g,j] := u[i,t] * u[j,t] * Cts[i,t,g]) ./= T-1
@test norm(err_ig2 - Array(err_ig)) < 1e-12

ζ, converged = estimate_giv(qmat, Cpts, Cts, Smat, zeros(Bool, N, N), Val{:iv}(); guess=ones(7))
@assert converged
@test norm(GIV.mean_vcov_err(
    ζ,
    Cqq,
    CqCp,
    CCpq,
    CCpCp,
    qq,
    Cpq,
    CpCp,
    Smat[:, 1],
    zeros(Bool, N, N),
)) < 1e-8

##================== test estimate_giv using vcov ==================##
# ζ2, converged = GIV.estimate_giv_with_vcov(qmat, Cpts, Cts, Smat; guess = ones(7))



##====== performance benchmark

# using Test, GIV
# using DataFrames, CSV
# # Random.seed!(6)
# simmodel = GIV.SimModel(T = 60, N = 10, varᵤshare = 0.8, usupplyshare = 0.2, h = 0.3, σᵤcurv = 0.2, ζs = 0.5, NC = 2, M = 0.5)
# # df = DataFrame(simmodel.data)
# # CSV.write("$(@__DIR__)/../examples/simdata1.csv", df)
# df = CSV.read("$(@__DIR__)/../examples/simdata1.csv", DataFrame)

# #============== assuming homogeneous elasticity ==============#
# givmodel = giv(df, @formula(q + endog(p) ~ id & (η1 + η2)), :id, :t, :absS; guess = Dict("Aggregate" => 2.0), algorithm = :scalar_search, dual_coef = true)

# f = @formula(q + endog(p) ~ id & (η1 + η2))
# response_term, slope_terms, endog_term, exog_terms = GIV.parse_endog(f)


# # est = estimate_model(simmodel.data, coefmapping = ones(Bool, 5, 1), ζSguess = 2.0,)
# # 1/est.M[1]
# @test givmodel.coef[1] * 2 ≈ 2.5341730 atol = 1e-4
# # delta_method(θ̂->[abs.(simmodel.data.S[:, 1])' * θ̂], est.ζ, est.Σζ)
# @test stderror(givmodel.coefdf[1, "p_coef"] * 2, givmodel) ≈ 0.2407 atol = 1e-4

# factor_coef = [0.2419 1.1729; 0.1842 0.3722; -1.3213 -0.6487; 0.6288 -0.4422; 0.7269 1.6341]
# @test givmodel.factor_coef ≈ vec(factor_coef) atol = 1e-4

# # algorithm uu
# @time givmodel_uu = giv(df, @formula(q + endog(p) ~ id & (η1 + η2)), :id, :t, :absS; guess = Dict("Constant" => 1.0), algorithm = :iv, dual_coef = true)
# @test givmodel_uu.coef ≈ givmodel.coef atol = 1e-6

# # algorithm vcov
# @time givmodel_uu = giv(df, @formula(q + endog(p) ~ id & (η1 + η2)), :id, :t, :absS; guess = Dict("Constant" => 1.0), algorithm = :iv_vcov, dual_coef = true)
