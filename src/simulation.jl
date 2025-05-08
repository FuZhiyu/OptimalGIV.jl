
@with_kw struct SimParam{TD<:Distribution{Univariate}}
    @deftype Float64
    γ = 0.5
    h = 0.2
    tailparam = 1.0 # to be updated during initialization
    M = 0.5 # targeted multiplier
    T::Int64 = 60
    K::Int64 = 0
    N::Int64 = 20
    ν = Inf
    NC::Int64 = 2
    varᵤshare = NC == 0 && K == 0 ? 1.0 : 0.2
    constS::Vector{Float64} = solve_S_for_hhi(h, N)
    σᵤcurv = 0.1
    σᵤvec::Vector{Float64} = specify_σᵢ(constS, σᵤcurv, 1.0, usupplyshare)
    DIST::TD = isinf(ν) ? Normal(0, 1.0) : TDist(ν) * sqrt((ν-2)/ν) * 1.0
    σp = 2.0
    σζ = 1.0
end

"""
To be backward compatible: allow for passing in NamedTuple
"""
function SimParam(tp::NamedTuple)
    return SimParam(;tp...)
end

"""
with positive curvature, idiosyncratic shocks are less volatile for larger entities
"""
function specify_σᵢ(S, curv, scale, usupplyshare)
    σᵢ² = exp.(-curv * log.(S))
    b = (S' * S) / (S' * Diagonal(σᵢ²) * S)
    σᵢ² .*= b
    σᵢ = sqrt.(σᵢ²) * scale
    if usupplyshare > 0
        σD² = sum((S .* σᵢ).^2) # demand side volatility
        σs² = usupplyshare * σD² / (1 - usupplyshare) # supply side volatility
        σs = sqrt(σs²)
        σᵢ = [σᵢ; σs]
    end
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
    S::Matrix{T}
    u::Matrix{T}
    m::Matrix{T}
    C::Matrix{T}
    Λ::Matrix{T}
    η::Matrix{T}
    p::Matrix{T}
    ζ::Vector{T}
    q::Matrix{T}
end

function DataFrame(simdata::SimData)
    N, T = size(simdata.q)
    df = DataFrame(
        S = vec(simdata.S),
        u = vec(simdata.u),
        q = vec(simdata.q),
        id = repeat(1:N, outer = T),
        ζ = repeat(simdata.ζ, outer = T),
        t = repeat(1:T, inner = N),
        p = repeat(vec(simdata.p),inner = N)
    )
    dfc = DataFrame(repeat(simdata.C', inner = (N, 1)), ["η$i" for i in 1:size(simdata.C, 1)])
    df = hcat(df, dfc)
    df.S = abs.(df.S)
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
    return SimData(;tpdict...)
end

struct SimModel
    param::SimParam
    data::SimData
end


function SimModel(;kwargs...)
    param = SimParam(;kwargs...)
    @unpack_SimParam param

    ζ = rand(Normal(0, σζ), N)
    ζ .-= sum(ζ .* constS)


    u = rand(DIST, N, T) .* σᵤvec
    m = rand(N, NC)
    C = rand(Normal(), NC, T)
    # normalize C: decorr and scale
    if NC > 0
        cholL = cholesky(cov(C')).L
        C = inv(cholL) * C
    end
    Λ = rand(N, K)
    η = rand(Normal(), K, T)
    
    if NC > 0 || K > 0
        commonshocks = m * C + Λ * η
        currentratio = var(constS' * u) / var(constS' * commonshocks)
        scaleidio = sqrt(varᵤshare / (currentratio * (1-varᵤshare)))
        u = u * scaleidio
        σᵤvec .= σᵤvec * scaleidio
    else
        @assert varᵤshare == 1.0
    end
    shock = u + m * C + Λ * η


    netshock = constS' * shock
    netshockscale = sqrt(var(netshock * M, mean = zero(eltype(netshock))) / σp^2)
    σᵤvec .= σᵤvec / netshockscale
    netshock = netshock / netshockscale
    u ./= netshockscale
    m ./= netshockscale
    Λ ./= netshockscale
    shock = u + m * C + Λ * η
    netshock = constS' * shock    
    p = reshape(netshock * M, 1, T) |> Matrix
    q = shock - ζ .* p
    
    S = constS * ones(1, T)
    simdata = SimData(S, u, m, C, Λ, η, p, ζ, q)

    model = SimModel(param, simdata)
    return model
end
