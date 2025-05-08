using Test, GIV, DataFrames, StatsBase, LinearAlgebra
GIV.Random.seed!(6)
simmodel = GIV.SimModel(T=60, N=4, varᵤshare=0.8, usupplyshare=0.2, h=0.3, σᵤcurv=0.2, ζs=0.5, NC=2, M=0.5, missingperc=0.1)
df = DataFrame(simmodel.data)

givmodel = giv(
    df, @formula(q + endog(p) ~ 0), :id, :t, :absS; algorithm=:iv_legacy, return_vcov=false)


function monte_carlo(;
    formula=@formula(q + endog(p) ~ 0),
    simulation_parameters=(;),
    estimation_parameters=(;),
)
    simmodel = GIV.SimModel(; simulation_parameters...)
    df = DataFrame(simmodel.data)
    givmodel = giv(df, formula, :id, :t, :absS; estimation_parameters...)
    bias = givmodel.coef - unique(vec(simmodel.data.ζ))
    ci = confint(givmodel)
    covered = ci[:, 1] .<= simmodel.data.ζ .<= ci[:, 2]
    if simulation_parameters.NC == 0
        return bias, covered
    else
        true_factor_coef = vec(simmodel.data.m)
        factor_bias = givmodel.factor_coef - true_factor_coef
        factor_se = sqrt.(diag(givmodel.factor_vcov))
        factor_ci = givmodel.factor_coef .+ factor_se .* reshape([-1.96, 1.96], 1, 2)
        factor_covered = factor_ci[:, 1] .<= vec(true_factor_coef) .<= factor_ci[:, 2]
        return bias, covered, factor_bias, factor_covered
    end
end

function monte_carlo(Nsims; seed=nothing, kwargs...)
    if !isnothing(seed)
        GIV.Random.seed!(seed)
    end

    biasvec = Vector{Union{Vector{Float64},Missing}}(undef, Nsims)
    coveredvec = Vector{Union{Vector{Bool},Missing}}(undef, Nsims)
    factor_biasvec = Vector{Union{Vector{Float64},Missing}}(undef, Nsims)
    factor_coveredvec = Vector{Union{Vector{Bool},Missing}}(undef, Nsims)
    for i in 1:Nsims
        # try 
        tup = monte_carlo(; kwargs...)
        bias = tup[1]
        covered = tup[2]
        biasvec[i] = bias
        coveredvec[i] = covered
        if length(tup) > 2
            factor_bias = tup[3]
            factor_covered = tup[4]
            factor_biasvec[i] = factor_bias
            factor_coveredvec[i] = factor_covered
        else
            factor_biasvec[i] = missing
            factor_coveredvec[i] = missing
        end
        # catch
        #     biasvec[i] = missing
        #     coveredvec[i] = missing
        #     factor_biasvec[i] = missing
        #     factor_coveredvec[i] = missing
        # end
    end
    biasvec = filter(!ismissing, biasvec)
    coveredvec = filter(!ismissing, coveredvec)
    stable = norm.(biasvec) .<= 100
    biasvec, coveredvec = biasvec[stable], coveredvec[stable]
    if !all(ismissing, factor_biasvec)
        factor_biasvec = filter(!ismissing, factor_biasvec)
        factor_coveredvec = filter(!ismissing, factor_coveredvec)
        factor_biasvec, factor_coveredvec = factor_biasvec[stable], factor_coveredvec[stable]
        return biasvec, coveredvec, factor_biasvec, factor_coveredvec
    else
        return biasvec, coveredvec
    end
end
simparams = (; T=100, N=5, varᵤshare=1, usupplyshare=0.0, h=0.2, σᵤcurv=0.1, ζs=0.0, NC=0, M=2, missingperc=0.01)
estparams = (;
    # guess=Dict("Aggregate" => 2.0),
    guess=ones(5),
    algorithm=:iv_legacy,
    quiet=true,
    return_vcov=false,
)
bias, covered = monte_carlo(100; seed=1,
    formula=@formula(q + id & endog(p) ~ 0),
    simulation_parameters=simparams,
    estimation_parameters=estparams
)

simmodel = GIV.SimModel(T=3000, N=2000, varᵤshare=1, usupplyshare=0.0, h=0.2, σᵤcurv=0.1, ζs=0.0, NC=0, M=1.0, σζ=0.0, missingperc=0.9)
df = DataFrame(simmodel.data)
# Save the DataFrame to a CSV file
using CSV, Arrow
CSV.write("simulated_data.csv", df)
giv(df, @formula(q + endog(p) ~ 0), :id, :t, :absS; algorithm=:iv, return_vcov=false, guess=[1.0]).coef[1]


simmodel = GIV.SimModel(T=300, N=200, varᵤshare=1, usupplyshare=0.0, h=0.2, σᵤcurv=0.1, ζs=0.0, NC=0, M=1.0, σζ=0.0, missingperc=0.1)
df = DataFrame(simmodel.data)
# Save the DataFrame to a CSV file
using CSV, Arrow
CSV.write("simulated_data_smaller.csv", df)
giv(df, @formula(q + endog(p) ~ 0), :id, :t, :absS; algorithm=:iv, return_vcov=false, guess=[1.0]).coef[1]


mean(mean.(bias))
@test mean(mean.(bias)) < 0.05

simparams = (; T=3000, N=2000, varᵤshare=1, usupplyshare=0.0, h=0.2, σᵤcurv=0.1, ζs=0.0, NC=0, M=1.0, σζ=0.0, missingperc=0.9)
estparams = (;
    guess=Dict("Constant" => 1.0),
    algorithm=:iv,
    quiet=true,
    return_vcov=false,
)

# bias, covered = monte_carlo(400; seed = 1,
bias, covered = monte_carlo(20;
    formula=@formula(q + endog(p) ~ 0),
    simulation_parameters=simparams,
    estimation_parameters=estparams
)

bias
mean(mean.(bias))

@test mean(mean.(bias)) < 0.05
