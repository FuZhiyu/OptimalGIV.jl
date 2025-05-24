using GIV, DataFrames, CSV, CategoricalArrays, JLD2, Statistics
using GIV: evaluation_metrics
using DataFramesMeta
Nsimmax = 100
simparamdf = CSV.read("simulations/simparamstr.csv", DataFrame)

function estimate_simulated_model(df::DataFrame, formula;
    guess=nothing,
    save=:none,
    quiet=true,
    solver_options=(; ftol=1e-4, iterations=100,),
    kwargs...)
    df.id = CategoricalArray(df.id)
    if isnothing(guess)
        guess = unique(df, :id).ζ
    end
    model = giv(df, formula, :id, :t, :S; guess=guess, save=save, quiet=quiet, solver_options=solver_options, kwargs...)
    return model
end

function estimate_and_evaluate(file, formula; kwargs...)
    df = CSV.read(file, DataFrame)
    return estimate_and_evaluate(df, formula; kwargs...)
end

function estimate_and_evaluate(df::DataFrame, formula; kwargs...)
    df.id = CategoricalArray(df.id)
    model = estimate_simulated_model(df, formula; kwargs...)
    return GIV.evaluation_metrics(model, df)
end

function estimate_simulated_models(simlabel, simparamstr, formula; Nsims=100, estimate_label="estimate", verbose=true, kwargs...)
    simpath = joinpath("simulations", simparamstr)
    simdata_files = readdir(simpath)
    simdata_files = joinpath.(simpath, filter(x -> occursin("simdata_", x), simdata_files))
    simdata_files = simdata_files[1:Nsims]

    metricdf = DataFrame(ζ_bias=Float64[], ζ_se=Float64[], β_bias=Float64[], β_se=Float64[])
    for i in 1:Nsims
        if verbose
            if i % 10 == 0
                println("[$simlabel] [$estimate_label] Estimating simulation $i of $Nsims")
            end
        end
        try
            est_metric = estimate_and_evaluate(simdata_files[i], formula; kwargs...)
            push!(metricdf, est_metric; promote=true)
        catch e
            println("Error in simulation $(simdata_files[i])")
            throw(e)

        end
    end
    metricdf.simulated_model .= simlabel
    metricdf.estimate_label .= estimate_label
    return metricdf
end


function summarize_metrics(metricdf)
    sdf = dropmissing(metricdf)
    ci_covered = (bias, se) -> mean(abs.(bias ./ se) .< 1.96)
    return combine(sdf,
        [:ζ_bias, :ζ_se, :β_bias, :β_se] .=> mean .=> (x -> x * "_mean"),
        [:ζ_bias, :ζ_se, :β_bias, :β_se] .=> median .=> (x -> x * "_median"),
        [:ζ_bias, :ζ_se] => ci_covered => "ζ_covered",
        [:β_bias, :β_se] => ci_covered => "β_covered",
        [:ζ_bias, :β_bias] .=> std => (x -> x * "_std"))
end
metricdf = mapreduce(vcat, eachrow(simparamdf[3:3, :])) do row
    estimate_simulated_models(row..., @formula(q + id & endog(p) ~ 0 + id & (η1 + η2)), estimate_label="standard", Nsims=100)
end

sparse_df = mapreduce(vcat, eachrow(simparamdf[5:5, :])) do row
    estimate_simulated_models(row..., @formula(q + endog(p) ~ 0 + fe(id) & (η1 + η2)), estimate_label="standard", Nsims=400, guess=[2.0], min_occurrences=2)
end

sparse_df_2 = mapreduce(vcat, eachrow(simparamdf[5:5, :])) do row
    estimate_simulated_models(row..., @formula(q + endog(p) ~ 0 + fe(id) & (η1 + η2)), estimate_label="min_occurrences=10", Nsims=400, guess=[2.0], min_occurrences=5)
end

sparse_df = vcat(sparse_df, sparse_df_2)
sparse_summary = combine(summarize_metrics, groupby(sparse_df, [:simulated_model, :estimate_label]))
vscodedisplay(sparse_summary)
vscodedisplay(sparse_df)
summary_df = combine(summarize_metrics, groupby(metricdf_uniform, [:simulated_model, :estimate_label]))
vscodedisplay(summary_df)

sparse_df = dropmissing(@subset(metricdf_uniform, :simulated_model .== "sparse panel"))
using Plots



simparamstr = simparamdf[5, :simparamstr]
simpath = joinpath("simulations", simparamstr)
simdata_files = readdir(simpath)
simdata_files = joinpath.(simpath, filter(x -> occursin("simdata_", x), simdata_files))
simdata_files = simdata_files[1:100]

df = CSV.read(simdata_files[13], DataFrame)
m = estimate_and_evaluate(df, @formula(q + endog(p) ~ 0 + fe(id) & (η1 + η2)), guess=[2.0])
m = estimate_simulated_model(df, @formula(q + endog(p) ~ 0 + fe(id) & (η1 + η2)), guess=[2.0])


q, Cp, C, uq, uCp, S, obs_index = GIV.extract_matrices(df, @formula(q + endog(p) ~ 0 + fe(id) & (η1 + η2)), :id, :t, :S)
σu²vec, Σζ = GIV.solve_vcov(uq, S, C, Cp, obs_index)
M = obs_index.entity_obs_indices .> 0
co = M * transpose(M)


function solve_vcov(u, S, C, Cp, obs_index)
    Nmom = size(C, 2)
    N, T = obs_index.N, obs_index.T
    σu²vec = GIV.calculate_entity_variance(u, obs_index)

    # Step 1: Efficiently identify all unique entity pairs that co-occur
    # 1) Build presence‐matrix
    M = obs_index.entity_obs_indices .> 0   # N×T BitMatrix
    # 2) Compute co‐occurrence counts
    co = M * transpose(M)                   # N×N dense Int matrix
    # 3) findall gives you CartesianIndex's for each true entry
    inds = findall(triu(co .> zero(eltype(co)), 1))       # CartesianIndices of every non-zero count in the upper triangle
    pair_i = [I[1] for I in inds]
    pair_j = [I[2] for I in inds]
    # Number of unique entity pairs
    n_pairs = length(pair_i)

    # Step 2: Initialize arrays for computation
    Vdiag = zeros(n_pairs)
    W = zeros(n_pairs, Nmom, T)
    D = zeros(n_pairs, Nmom, T)

    # Step 3: Compute Vdiag for all pairs
    for idx in 1:n_pairs
        i, j = pair_i[idx], pair_j[idx]
        Vdiag[idx] = σu²vec[i] * σu²vec[j]
    end

    precision = 1 ./ σu²vec
    # Step 4: Compute W matrix (previous D without Mvec scaling)
    for t in 1:T
        for idx in 1:n_pairs
            i, j = pair_i[idx], pair_j[idx]
            i_pos = obs_index.entity_obs_indices[i, t]
            j_pos = obs_index.entity_obs_indices[j, t]
            if i_pos == 0 || j_pos == 0
                continue
            end
            for k in 1:Nmom
                # when we do not have the full market, we do not scale it by Mvec
                W[idx, k, t] = precision[i] * S[j_pos] * C[i_pos, k] + precision[j] * S[i_pos] * C[j_pos, k]
                D[idx, k, t] = u[j_pos] * Cp[i_pos, k] + u[i_pos] * Cp[j_pos, k]
            end
        end
    end

    # Step 6: Final calculation via sandwich formula
    # Compute sums of D'W and W'Vdiag W across time periods
    A = zeros(eltype(u), Nmom, Nmom)
    B = zeros(eltype(u), Nmom, Nmom)
    @views for t in 1:T
        Wt = W[:, :, t]
        Dt = D[:, :, t]

        A .+= Dt' * Wt
        B .+= Wt' * (Wt .* Vdiag)
    end
    A ./= (T - 1) # to match the solve_vcov as the σu²vec is scaled by T-1
    B ./= T
    B = Symmetric(B + B') / 2
    # Sandwich variance
    invA = inv(A)
    Σζ = invA * B * invA' / T
    Σζ = Symmetric(Σζ + Σζ') / 2
    return A, B, σu²vec, Σζ
end
using LinearAlgebra
A, B, residual_variance, Σζ = solve_vcov(uq, S, C, Cp, obs_index)
A
B

residual_variance
findall(obs_index.entity_obs_indices[end, :] .> 0)

Σζ[1, 1]
inds = findall(triu(co .> zero(eltype(co)), 1))
pair_i = [I[1] for I in inds]
pair_j = [I[2] for I in inds]




evaluation_metrics(m, df)
m.vcov
m.Nelasticities
m = estimate_simulated_model(df, @formula(q + id & endog(p) ~ 0 + id & (η1 + η2)), complete_coverage=true)
# GIV.evaluation_metrics(m, df)
m.coef[11:20]
m2 = estimate_simulated_model(df, @formula(q + id & endog(p) ~ 0 + fe(id) & (η1 + η2)), save=:fe, complete_coverage=true)
sort!(m2.coefdf, :id)

m.coef[11:20] - m2.coefdf[!, "fe_id&η1"]



m.vcov[1:10, 1:10]
m2.vcov[1:10, 1:10]


df = CSV.read("simulations/N=10_T=100_K=2_ushare=0.5_σζ=1.0_missingperc=0.5/simdata_111.csv", DataFrame)
m = estimate_simulated_model(df, @formula(q + id & endog(p) ~ 0), complete_coverage=false, save=:all)

u = m.df.q_residual
diag
Vhat = (m.vcov + m.vcov') / 2        # force symmetry
λ = eigvals(Vhat)
isposdef(Vhat)
Σ = m.vcov
Σ - Σ'
Σ

@show minimum(λ), sum(λ .< -1e-8), sum(abs.(λ) .< 1e-12)

for i in 100:120
    try
        df = CSV.read("simulations/N=10_T=100_K=2_ushare=0.5_σζ=1.0_missingperc=0.5/simdata_$i.csv", DataFrame)
        m = estimate_simulated_model(df, @formula(q + id & endog(p) ~ 0), complete_coverage=false, save=:all)
        sqrt.(diag(m.vcov))
    catch e
        println("Error in simulation $i")
        throw(e)
    end
end

diag(m.vcov)
chol = cholesky(Symmetric(m.vcov))


vscodedisplay(m.df)
using DataFramesMeta

@subset(m.df, :id .== 9)
propertynames(m)
m.residual_variance

using LinearAlgebra
sqrt.(diag(m.vcov))
S = unique(df, :id).S
S' * m.vcov * S

fullcovdf = estimate_simulated_models(simparamdf[3, :]..., @formula(q + id & endog(p) ~ 0 + id & (η1 + η2)), estimate_label="full_coverage", Nsims=100, complete_coverage=true)

metricdf = vcat(nocovdf, fullcovdf)
summary_df = combine(summarize_metrics, groupby(metricdf, [:simulated_model, :estimate_label]))
vscodedisplay(summary_df)
summarize_metrics(nocovdf)

metricdf = mapreduce(vcat, eachrow(simparamdf[1:2, :])) do row
    estimate_simulated_models(row..., @formula(q + id & endog(p) ~ 0 + fe(id) & (η1 + η2)), estimate_label="standard", Nsims=100)extra
end

summary_df = combine(summarize_metrics, groupby(metricdf, [:simulated_model, :estimate_label]))
