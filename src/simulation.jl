@with_kw struct SimParam{TD<:Distribution{Univariate}}
    @deftype Float64
    h = 0.2
    tailparam = 1.0 # to be updated during initialization
    M = 0.5 # targeted multiplier
    T::Int64 = 100
    K::Int64 = 2
    N::Int64 = 10
    ν = Inf
    ushare = K == 0 ? 1.0 : 0.2
    constS::Vector{Float64} = solve_S_for_hhi(h, N)
    σᵤcurv = 0.1
    σᵤvec::Vector{Float64} = specify_σᵢ(constS, σᵤcurv, 1.0, ushare)
    DIST::TD = isinf(ν) ? Normal(0, 1.0) : TDist(ν) * sqrt((ν - 2) / ν) * 1.0
    σp = 2.0
    σζ = 1.0
    missingperc = 0.0 # percentage of missing values in the data
end

"""
To be backward compatible: allow for passing in NamedTuple
"""
function SimParam(tp::NamedTuple)
    return SimParam(; tp...)
end

"""
with positive curvature, idiosyncratic shocks are less volatile for larger entities
"""
function specify_σᵢ(S, curv, scale, usupplyshare)
    σᵢ² = exp.(-curv * log.(S))
    b = (S' * S) / (S' * Diagonal(σᵢ²) * S)
    σᵢ² .*= b
    σᵢ = sqrt.(σᵢ²) * scale
    return σᵢ
end

"""
Given an excessive HHI, solve the tail parameter to get the size distribution
"""
function solve_S_for_hhi(h, N)
    function h_from_tailparam(tailparam)
        k = [1:N;] .^ (-1 / tailparam)
        S = k / sum(k)
        h′ = sqrt(sum(S .^ 2) - 1 / N)
        return h - h′
    end
    tailparam = find_zero(h_from_tailparam, 1.0)
    k = [1:N;] .^ (-1 / tailparam)
    S = k / sum(k)
    return S
end

@with_kw struct SimData{T}
    S::Matrix{Union{T,Missing}}
    u::Matrix{Union{T,Missing}}
    Λ::Matrix{Union{T,Missing}}
    η::Matrix{Union{T,Missing}}
    p::Matrix{Union{T,Missing}}
    ζ::Vector{Union{T,Missing}}
    q::Matrix{Union{T,Missing}}
end

function DataFrames.DataFrame(simdata::SimData)
    N, T = size(simdata.q)
    df = DataFrame(
        S=vec(simdata.S),
        u=vec(simdata.u),
        q=vec(simdata.q),
        id=repeat(string.(1:N), outer=T),
        ζ=repeat(simdata.ζ, outer=T),
        t=repeat(1:T, inner=N),
        p=repeat(vec(simdata.p), inner=N)
    )
    dfη = DataFrame(repeat(simdata.η', inner=(N, 1)), ["η$i" for i in 1:size(simdata.η, 1)])
    dfλ = DataFrame(repeat(simdata.Λ, outer=(T, 1)), ["λ$i" for i in 1:size(simdata.Λ, 2)])
    df = hcat(df, dfη, dfλ)
    dropmissing!(df, :q)
    disallowmissing!(df)
    return df
end

"""
To be backward compatible: allow for passing in NamedTuple
"""
function SimData(tp::NamedTuple)
    # convert NamedTuple to dict
    tpdict = ntuple2dict(tp)

    # remove those keys that are not in the struct
    for key in keys(tpdict)
        if !(Symbol(key) in fieldnames(SimData))
            delete!(tpdict, key)
        end
    end
    return SimData(; tpdict...)
end

struct SimModel
    param::SimParam
    data::SimData
end


function SimModel(; kwargs...)
    param = SimParam(; kwargs...)
    @unpack_SimParam param

    ζ = rand(Normal(0, σζ), N)
    ζ .-= sum(ζ .* constS)
    ζ .+= 1 / M

    u = rand(DIST, N, T) .* σᵤvec
    # normalize C: decorr and scale
    if K > 0
        η = rand(Normal(), K, T)
        # cholL = cholesky(cov(η')).L
        # η = inv(cholL) * η
        Λ = rand(N, K)
    end

    if K > 0
        commonshocks = Λ * η
        currentratio = var(constS' * u) / var(constS' * commonshocks)
        scaleidio = sqrt(ushare / (currentratio * (1 - ushare)))
        u = u * scaleidio
        σᵤvec .= σᵤvec * scaleidio
    else
        @assert ushare == 1.0
    end
    shock = u + Λ * η

    netshock = constS' * shock
    netshockscale = sqrt(var(netshock * M, mean=zero(eltype(netshock))) / σp^2)
    σᵤvec .= σᵤvec / netshockscale
    netshock = netshock / netshockscale
    u ./= netshockscale
    Λ ./= netshockscale
    shock = u + Λ * η
    netshock = constS' * shock
    p = reshape(netshock * M, 1, T) |> Matrix
    q = shock - ζ .* p

    # Convert all matrices to allow missing values
    S = allowmissing(constS * ones(1, T))
    u = allowmissing(u)
    Λ = allowmissing(Λ)
    η = allowmissing(η)
    p = allowmissing(p)
    ζ = allowmissing(ζ)
    q = allowmissing(q)

    # Introduce missingness
    if missingperc > 0
        missing_mask = rand(N, T) .< missingperc
        q[missing_mask] .= missing
    end

    simdata = SimData(S, u, Λ, η, p, ζ, q)

    model = SimModel(param, simdata)
    return model
end


function evaluation_metrics(m::GIVModel, df)
    if !m.converged
        return [missing, missing, missing, missing]
    end

    coefnames = m.coefnames
    # estparam = m.coef
    iddf = sort(unique(df, :id), :id)
    trueζ = iddf[!, :ζ]
    estζ = m.coef[1:m.Nelasticities]
    Σζ = m.vcov[1:m.Nelasticities, 1:m.Nelasticities]
    if m.Nelasticities == 1
        S = [1.0]
        trueζ = trueζ[1:1]
    else
        S = iddf.S
    end
    # se_ζ = sqrt.(diag(Σζ))
    # bias = mean(estζ - trueζ)
    # covered = mean(abs.(estζ - trueζ) .<= se_ζ * 1.96)

    trueaggζ = sum(S .* trueζ)
    estaggζ = sum(S .* estζ)
    Σaggζ = S' * Σζ * S
    se_aggζ = sqrt(Σaggζ[1])
    biasaggζ = estaggζ - trueaggζ
    coveredaggζ = abs(estaggζ - trueaggζ) <= se_aggζ * 1.96

    if length(m.coef) == m.Nelasticities
        return [biasaggζ, se_aggζ, NaN, NaN]
    end
    estβ = m.coef[m.Nelasticities+1:end]
    Σβ = m.vcov[m.Nelasticities+1:end, m.Nelasticities+1:end]
    K = length(names(iddf, r"η"))
    trueβ = Float64[]
    for k in 1:K
        if any(occursin.("η$k", coefnames))
            trueβ = vcat(trueβ, iddf[!, Symbol("λ$k")])
        end
    end
    biasβ = mean(estβ - trueβ)
    se_β = sqrt.(diag(Σβ))
    N = length(iddf.id)
    coveredβ = mean(abs.(estβ - trueβ) .<= se_β * 1.96)
    estimatedK = length(trueβ) ÷ N
    Sβ = repeat(S, outer=estimatedK) / estimatedK
    trueaggβ = sum(Sβ .* trueβ)
    estaggβ = sum(Sβ .* estβ)
    Σaggβ = Sβ' * Σβ * Sβ
    se_aggβ = sqrt(Σaggβ[1])
    biasaggβ = estaggβ - trueaggβ
    coveredaggβ = abs(estaggβ - trueaggβ) <= se_aggβ * 1.96

    return [biasaggζ, se_aggζ, biasaggβ, se_aggβ]
end