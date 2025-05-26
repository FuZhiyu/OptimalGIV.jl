using Test, OptimalGIV, DataFrames, StatsBase, LinearAlgebra
OptimalGIV.Random.seed!(6)

simmodel = OptimalGIV.SimModel(T=60, N=4, varᵤshare=0.8, usupplyshare=0.2, h=0.3, σᵤcurv=0.2, ζs=0.5, NC=2, M=0.5)
df = DataFrame(simmodel.data)
# CSV.write("$(@__DIR__)/../examples/simdata1.csv", df)
# df = CSV.read("$(@__DIR__)/../examples/simdata1.csv", DataFrame)

function monte_carlo(;    
    formula = @formula(q + endog(p) ~ 0),
    simulation_parameters = (;),
    estimation_parameters = (;),
)
    simmodel = OptimalGIV.SimModel(; simulation_parameters...)
    df = DataFrame(simmodel.data)
    givmodel = giv(df,formula, :id, :t, :absS; estimation_parameters...)
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

function monte_carlo(Nsims; seed = nothing, kwargs...)
    if !isnothing(seed)
        OptimalGIV.Random.seed!(seed)
    end

    biasvec = Vector{Union{Vector{Float64}, Missing}}(undef, Nsims)
    coveredvec = Vector{Union{Vector{Bool}, Missing}}(undef, Nsims)
    factor_biasvec = Vector{Union{Vector{Float64},Missing}}(undef, Nsims)
    factor_coveredvec = Vector{Union{Vector{Bool},Missing}}(undef, Nsims)
    for i in 1:Nsims
        try 
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
        catch
            biasvec[i] = missing
            coveredvec[i] = missing
            factor_biasvec[i] = missing
            factor_coveredvec[i] = missing
        end
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
givmodel = giv(
    df,
    @formula(q + id & endog(p) ~ id & (η1 + η2)),
    :id,
    :t,
    :absS;
    guess = Dict("Aggregate" => 2.0),
    algorithm = :scalar_search,
    return_vcov = false,
)
#==============  homogeneous elasticity ==============#
simparams = (;T = 100, N = 10, varᵤshare = 1, usupplyshare = 0.0, h = 0.2, σᵤcurv = 0.1, ζs = 0.0, NC = 0, M = 0.5, σζ = 0.0)
estparams = (;
    guess = Dict("Aggregate" => 2.0),
    algorithm = :scalar_search,
    quiet = true,
)

bias, covered = monte_carlo(400; seed = 1,
    formula = @formula(q + endog(p) ~ 0),
    simulation_parameters = simparams,
    estimation_parameters = estparams
)
@test mean(mean.(bias)) < 0.05
@test mean(mean.(covered)) ≈ 0.95 atol = 0.01

#============== assuming heterogeneous elasticity ==============#
simparams = (;T = 100, N = 10, varᵤshare = 1, usupplyshare = 0.0, h = 0.2, σᵤcurv = 0.1, ζs = 0.0, NC = 0, M = 0.5)
estparams = (;
    guess = Dict("Aggregate" => 2.0),
    algorithm = :scalar_search,
    quiet = true,
)
bias, covered = monte_carlo(400; seed = 1,
    formula = @formula(q + id & endog(p) ~ 0),
    simulation_parameters = simparams,
    estimation_parameters = estparams
)
@test mean(mean.(bias)) < 0.05
@test mean(mean.(covered)) ≈ 0.95 atol = 0.02 # relax a bit

##================ test se for factor loadings  ==================##
simparams = (;
    T = 100,
    N = 10,
    varᵤshare = 0.3,
    usupplyshare = 0.0,
    h = 0.2,
    σᵤcurv = 0.1,
    ζs = 0.0,
    NC = 3,
    M = 0.5,
)
estparams = (; guess = Dict("Aggregate" => 2.0), algorithm = :scalar_search, quiet = true)
bias, covered, factor_bias, factor_covered = monte_carlo(
    400,
    ;
    seed = 1,
    formula = @formula(q + id & endog(p) ~ id & (η1 + η2 + η3)),
    simulation_parameters = simparams,
    estimation_parameters = estparams,
)

@test mean(mean.(covered)) ≈ 0.95 atol = 0.1 # relax a bit
mean(mean.(factor_covered))
@test mean(mean.(factor_covered)) ≈ 0.95 atol = 0.015 # relax a bit
##================ test interaction ==================##
function add_interaction(simmodel; Δζ = -0.5)
    param, data = simmodel.param, simmodel.data
    OptimalGIV.@unpack_SimData data
    N, T = size(q)
    shock = u + m * C + Λ * η
    absS = abs.(S[:, 1])
    netshock = absS' * shock    
    ζS = absS' * ζ
    ζS2 = ζS + Δζ * sum(absS)
    secondhalf = 1:T .> T/2
    p[1, secondhalf] .= netshock[secondhalf] / ζS2
    q[:, secondhalf] .= shock[:, secondhalf] - (ζ .+ Δζ) .* p[1:1, secondhalf]
    @assert all(<(1e-8), sum(q .* absS, dims = 1))
    return simmodel
end

function monte_carlo_with_interaction(;    
    Δζ = -0.5,
    formula = @formula(q + endog(p) ~ 0),
    simulation_parameters = (;),
    estimation_parameters = (;),
)
    simmodel = OptimalGIV.SimModel(; simulation_parameters...)
    simmodel = add_interaction(simmodel; Δζ = Δζ)
    df = DataFrame(simmodel.data)
    df.secondhalf = df.t .> maximum(df.t)/2
    guess = Dict("id" => unique(df.ζ), "secondhalf" => Δζ)
    givmodel = giv(df,formula, :id, :t, :absS; guess = guess, estimation_parameters...)
    truecoef = [unique(df.ζ); Δζ]
    bias = givmodel.coef - truecoef
    ci = confint(givmodel)
    covered = ci[:, 1] .<= [unique(df.ζ); Δζ] .<= ci[:, 2]
    return bias, covered
end

function monte_carlo_with_interaction(Nsims; seed = nothing, kwargs...)
    if !isnothing(seed)
        OptimalGIV.Random.seed!(seed)
    end

    biasvec = Vector{Union{Vector{Float64}, Missing}}(undef, Nsims)
    coveredvec = Vector{Union{Vector{Bool}, Missing}}(undef, Nsims)
    for i in 1:Nsims
        try 
            bias, covered = monte_carlo_with_interaction(;kwargs...)
            biasvec[i] = bias
            coveredvec[i] = covered
        catch
            biasvec[i] = missing
            coveredvec[i] = missing
        end
    end
    biasvec = filter(!ismissing, biasvec)
    coveredvec = filter(!ismissing, coveredvec)
    stable = norm.(biasvec) .<= 100
    biasvec, coveredvec = biasvec[stable], coveredvec[stable]
    return biasvec, coveredvec
end

simparams = (;T = 100, N = 5, varᵤshare = 1, usupplyshare = 0.0, h = 0.2, σᵤcurv = 0.1, ζs = 0.0, NC = 0, M = 0.5)
estparams = (; algorithm=:iv, quiet=true)

bias, covered = monte_carlo_with_interaction(400; seed = 1,
    Δζ = -1,
    formula = @formula(q + id & endog(p) + secondhalf & endog(p)~ secondhalf),
    simulation_parameters = simparams,
    estimation_parameters = estparams
)

@test mean(last.(bias)) < 0.01
@test mean(last.(covered)) ≈ 0.95 atol = 0.1

