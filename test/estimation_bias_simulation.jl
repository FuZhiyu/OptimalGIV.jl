using Plots
using Test, GIV, DataFrames, StatsBase, LinearAlgebra
GIV.Random.seed!(6)
simmodel = GIV.SimModel(T=60, N=4, varᵤshare=0.8, usupplyshare=0.2, h=0.3, σᵤcurv=0.2, ζs=0.5, NC=2, M=0.5, missingperc=0.1)
# df = DataFrame(simmodel.data)
simmodel.param.M
# givmodel = giv(
#     df, @formula(q + endog(p) ~ 0), :id, :t, :absS; algorithm=:iv_legacy, return_vcov=false)


function monte_carlo(;
    formula=@formula(q + endog(p) ~ 0),
    simulation_parameters=(;),
    estimation_parameters=(;),
)
    simmodel = GIV.SimModel(; simulation_parameters...)
    df = DataFrame(simmodel.data)
    givmodel = giv(df, formula, :id, :t, :absS; estimation_parameters...)
    if !givmodel.converged
        return missing
    end
    bias = (1 / simmodel.param.M  - givmodel.agg_coef[1]) * simmodel.param.M
    return bias
end

function monte_carlo(Nsims; seed=nothing, kwargs...)
    if !isnothing(seed)
        GIV.Random.seed!(seed)
    end
    biasvec = Vector{Union{Missing,Float64}}(undef, Nsims)
    for i in 1:Nsims
        # try 
        bias = monte_carlo(; kwargs...)     
        biasvec[i] = bias
        # coveredvec[i] = covered
        # if length(tup) > 2
        #     factor_bias = tup[3]
        #     factor_covered = tup[4]
        #     factor_biasvec[i] = factor_bias
        #     factor_coveredvec[i] = factor_covered
        # else
        #     factor_biasvec[i] = missing
        #     factor_coveredvec[i] = missing
        # end
        # catch
        #     biasvec[i] = missing
        #     coveredvec[i] = missing
        #     factor_biasvec[i] = missing
        #     factor_coveredvec[i] = missing
        # end
    end
    biasvec = filter(!ismissing, biasvec)
    # coveredvec = filter(!ismissing, coveredvec)
    # stable = norm.(biasvec) .<= 100
    # biasvec, coveredvec = biasvec[stable], coveredvec[stable]
    # if !all(ismissing, factor_biasvec)
    #     factor_biasvec = filter(!ismissing, factor_biasvec)
    #     factor_coveredvec = filter(!ismissing, factor_coveredvec)
    #     factor_biasvec, factor_coveredvec = factor_biasvec[stable], factor_coveredvec[stable]
    #     return biasvec, coveredvec, factor_biasvec, factor_coveredvec
    # else
    #     return biasvec, coveredvec
    # end
end
# simparams = (; T=100, N=5, varᵤshare=1, usupplyshare=0.0, h=0.2, σᵤcurv=0.1, ζs=0.0, NC=0, M=1, missingperc=0.00)
# estparams = (;
#     # guess=Dict("Aggregate" => 2.0),
#     guess=ones(5),
#     algorithm=:iv_legacy,
#     quiet=true,
#     return_vcov=false,
# )
# bias = monte_carlo(100; seed=2,
#     formula=@formula(q + id & endog(p) ~ 0),
#     simulation_parameters=simparams,
#     estimation_parameters=estparams
# )
# mean(bias)


## symmetric case
# Loop over varᵤshare values and report biases
var_ushares = 0.95:-0.1:0.1
biases = Float64[]
estparams = (;
    # guess=Dict("Aggregate" => 2.0),
    guess=[1.0],
    algorithm=:iv_legacy,
    quiet=true,
    return_vcov=false,
)
for v in var_ushares
    # simparams = (; T=100, N=10, varᵤshare=v, usupplyshare=0.0, h=0.2, σᵤcurv=0.1, ζs=0.0, NC=1, M=1, missingperc=0.00)
    println("varᵤshare=$(v)")
    simparams = (; T=100, N=100, varᵤshare=v, usupplyshare=0.0, h=0.0, σᵤcurv=0.0, ζs=0.0, NC=3, M=1, missingperc=0.00, σζ = 0.0)
    bias = monte_carlo(100; seed=2,
        formula=@formula(q +  endog(p) ~ 0),
        simulation_parameters=simparams,
        estimation_parameters=estparams
    )
    push!(biases, mean(skipmissing(bias)))
    # println("varᵤshare=$(v): bias=$(biases[v])")
end
plot(1 .- var_ushares, biases, label="Simulated Bias")
plot!(1 .- var_ushares, 1 .- sqrt.(var_ushares), label="Approximation formula")
# println("Biases by varᵤshare: ", biases)

## with fat tailness now
# Loop over varᵤshare values and report biases
biases = Float64[]
for v in var_ushares
    # simparams = (; T=100, N=10, varᵤshare=v, usupplyshare=0.0, h=0.2, σᵤcurv=0.1, ζs=0.0, NC=1, M=1, missingperc=0.00)
    println("varᵤshare=$(v)")
    simparams = (; T=100, N=10, varᵤshare=v, usupplyshare=0.0, h=0.37, σᵤcurv=0.38, ζs=0.0, NC=1, M=1, missingperc=0.00, σζ = 0)
    bias = monte_carlo(100; seed=2,
        formula=@formula(q +  endog(p) ~ 0),
        simulation_parameters=simparams,
        estimation_parameters=estparams
    )
    push!(biases, mean(skipmissing(bias)))
    # println("varᵤshare=$(v): bias=$(biases[v])")
end
plot(1 .- var_ushares, biases, label="Simulated Bias")
plot!(1 .- var_ushares, 1 .- sqrt.(var_ushares), label="Approximation formula")
# println("Biases by varᵤshare: ", biases)


## larger N
biases = Float64[]
estparams = (;
    # guess=Dict("Aggregate" => 2.0),
    guess=[1.0],
    algorithm=:iv,
    quiet=true,
    return_vcov=false,
)
for v in var_ushares
    # simparams = (; T=100, N=10, varᵤshare=v, usupplyshare=0.0, h=0.2, σᵤcurv=0.1, ζs=0.0, NC=1, M=1, missingperc=0.00)
    println("varᵤshare=$(v)")
    simparams = (; T=100, N=100, varᵤshare=v, usupplyshare=0.0, h=0.2, σᵤcurv=0., ζs=0.0, NC=1, M=1, missingperc=0.00, σζ = 0.0)
    bias = monte_carlo(100; seed=2,
        formula=@formula(q +  endog(p) ~ 0),
        simulation_parameters=simparams,
        estimation_parameters=estparams
    )
    push!(biases, mean(skipmissing(bias)))
    # println("varᵤshare=$(v): bias=$(biases[v])")
end
plot(1 .- var_ushares, biases, label="Simulated Bias")
plot!(1 .- var_ushares, 1 .- sqrt.(var_ushares), label="Approximation formula")
# println("Biases by varᵤshare: ", biases)

## symmetric model but estimated with heterogeneous elasticity
biases = Float64[]
estparams = (;
    # guess=Dict("Aggregate" => 2.0),
    guess=ones(10),
    algorithm=:iv_legacy,
    quiet=true,
    return_vcov=false,
    solver_options= (; iterations = 100, ftol=1e-4),
)
for v in var_ushares
    # simparams = (; T=100, N=10, varᵤshare=v, usupplyshare=0.0, h=0.2, σᵤcurv=0.1, ζs=0.0, NC=1, M=1, missingperc=0.00)
    println("varᵤshare=$(v)")
    simparams = (; T=100, N=10, varᵤshare=v, usupplyshare=0.0, h=0.2, σᵤcurv=0.0, ζs=0.0, NC=1, M=1, missingperc=0.00, σζ = 0.0)

    bias = monte_carlo(100; seed=3,
        formula=@formula(q + id & endog(p) ~ 0),
        simulation_parameters=simparams,
        estimation_parameters=estparams
    )
    
    push!(biases, mean(trim(bias, prop = 0.05)))
    # println("varᵤshare=$(v): bias=$(biases[v])")
end
plot(1 .- var_ushares, biases, label="Simulated Bias")
plot!(1 .- var_ushares, 1 .- sqrt.(var_ushares), label="Approximation formula")

## further introduce heterogeneous elasticity
# Loop over varᵤshare values and report biases
biases = Float64[]
estparams = (;
    # guess=Dict("Aggregate" => 2.0),
    guess=ones(10),
    algorithm=:iv_legacy,
    quiet=true,
    return_vcov=false,
    solver_options= (; iterations = 100, ftol=1e-4),
)
for v in var_ushares
    # simparams = (; T=100, N=10, varᵤshare=v, usupplyshare=0.0, h=0.2, σᵤcurv=0.1, ζs=0.0, NC=1, M=1, missingperc=0.00)
    println("varᵤshare=$(v)")
    simparams = (; T=100, N=10, varᵤshare=v, usupplyshare=0.0, h=0.3, σᵤcurv=0, ζs=0.0, NC=1, M=1, missingperc=0.00, σζ = 1)

    bias = monte_carlo(100; seed=3,
        formula=@formula(q + id & endog(p) ~ 0),
        simulation_parameters=simparams,
        estimation_parameters=estparams
    )
    push!(biases, mean(trim(bias, prop = 0.05)))
    # println("varᵤshare=$(v): bias=$(biases[v])")
end
plot(1 .- var_ushares, biases, label="Simulated Bias")
plot!(1 .- var_ushares, 1 .- sqrt.(var_ushares), label="Approximation formula")
# println("Biases by varᵤshare: ", biases)


simparams = (; T=100, N=10, varᵤshare=0.5, usupplyshare=0.0, h=0.3, σᵤcurv=0.0, ζs=0.0, NC=1, M=1, missingperc=0.00, σζ = 0.0)
simmodel = GIV.SimModel(; simparams...)

df = DataFrame(simmodel.data)
givmodel1 = giv(df, @formula(q + id & endog(p) ~ 0), :id, :t, :absS;     
    guess= ones(10),
    algorithm=:iv_legacy,
    quiet=true,
    return_vcov=false,
    solver_options= (; iterations = 100, ftol=1e-4, show_trace=true),
)
givmodel2 = giv(df, @formula(q + endog(p) ~ 0), :id, :t, :absS; 
    guess= [1.0],
    algorithm=:iv_legacy,
    quiet=true,
    return_vcov=false,
    solver_options= (; iterations = 100, ftol=1e-4, show_trace=true),
)
## test the simulation more carefully

