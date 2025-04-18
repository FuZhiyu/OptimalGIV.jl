struct GIVModel <: StatisticalModel
    coef::Vector{Float64}
    vcov::Matrix{Float64}
    factor_coef::Vector{Float64}
    factor_vcov::Matrix{Float64}
    price_factor_coef::Matrix{Float64}
    residual_variance::Vector{Float64}
    agg_coef::Union{Float64,Vector{Float64}}

    formula::FormulaTerm
    responsename::String
    endogname::String
    coefnames::Vector{String}
    factor_coefnames::Vector{String}
    idvar::Symbol
    tvar::Symbol
    weightvar::Symbol
    exclude_pairs::Vector{Pair}

    coefdf::DataFrame
    df::Union{DataFrame,Nothing}

    converged::Bool
    N::Int64
    T::Int64
    nobs::Int64
    dof::Int64
    dof_residual::Int64
end

StatsAPI.coef(m::GIVModel) = m.coef
StatsAPI.coefnames(m::GIVModel) = m.coefnames
StatsAPI.responsename(m::GIVModel) = m.responsename
StatsAPI.vcov(m::GIVModel) = m.vcov
StatsAPI.nobs(m::GIVModel) = m.nobs
# StatsAPI.dof(m::GIVModel) = m.dof
StatsAPI.dof_residual(m::GIVModel) = m.dof_residual
# StatsAPI.r2(m::GIVModel) = r2(m, :devianceratio)
StatsAPI.islinear(m::GIVModel) = true
# StatsAPI.deviance(m::GIVModel) = rss(m)
# StatsAPI.nulldeviance(m::GIVModel) = m.tss
# StatsAPI.rss(m::GIVModel) = m.rss
# StatsAPI.mss(m::GIVModel) = nulldeviance(m) - rss(m)

StatsModels.formula(m::GIVModel) = m.formula
StatsAPI.residuals(m::GIVModel) = StatsAPI.residuals(m, m.df)

StatsAPI.residuals(m::GIVModel, ::Nothing) =
    throw(ArgumentError("DataFrame not saved. Rerun the model with `save_df = true`"))

StatsAPI.residuals(m::GIVModel, df) = df[!, Symbol(m.responsename, "_residual")]


function StatsAPI.confint(m::GIVModel; level::Real = 0.95)
    scale = tdistinvcdf(StatsAPI.dof_residual(m), 1 - (1 - level) / 2)
    se = stderror(m)
    return hcat(m.coef - scale * se, m.coef + scale * se)
end

function StatsAPI.coeftable(m::GIVModel; level = 0.95)
    cc = coef(m)
    se = stderror(m)
    coefnms = coefnames(m)
    conf_int = confint(m; level = level)
    # put (intercept) last
    # if !isempty(coefnms) && ((coefnms[1] == Symbol("(Intercept)")) || (coefnms[1] == "(Intercept)"))
    #     newindex = vcat(2:length(cc), 1)
    #     cc = cc[newindex]
    #     se = se[newindex]
    #     conf_int = conf_int[newindex, :]
    #     coefnms = coefnms[newindex]
    # end
    tt = cc ./ se
    return CoefTable(
        hcat(
            cc,
            se,
            tt,
            fdistccdf.(Ref(1), Ref(StatsAPI.dof_residual(m)), abs2.(tt)),
            conf_int[:, 1:2],
        ),
        ["Estimate", "Std. Error", "t-stat", "Pr(>|t|)", "Lower 95%", "Upper 95%"],
        ["$(coefnms[i])" for i in 1:length(cc)],
        4,
    )
end

import StatsBase: NoQuote, PValue
function Base.show(io::IO, m::GIVModel)
    ct = coeftable(m)
    #copied from show(iio,cf::Coeftable)
    cols = ct.cols
    rownms = ct.rownms
    colnms = ct.colnms
    nc = length(cols)
    nr = length(cols[1])
    if length(rownms) == 0
        rownms = [lpad("[$i]", floor(Integer, log10(nr)) + 3) for i in 1:nr]
    end
    mat = [
        j == 1 ? NoQuote(rownms[i]) :
        j - 1 == ct.pvalcol ? NoQuote(sprint(show, PValue(cols[j-1][i]))) :
        j - 1 in ct.teststatcol ? TestStat(cols[j-1][i]) :
        cols[j-1][i] isa AbstractString ? NoQuote(cols[j-1][i]) : cols[j-1][i] for i in 1:nr,
        j in 1:nc+1
    ]
    io = IOContext(io, :compact => true, :limit => false)
    A = Base.alignment(io, mat, 1:size(mat, 1), 1:size(mat, 2), typemax(Int), typemax(Int), 3)
    nmswidths = pushfirst!(length.(colnms), 0)
    A = [
        nmswidths[i] > sum(A[i]) ? (A[i][1] + nmswidths[i] - sum(A[i]), A[i][2]) : A[i] for
        i in 1:length(A)
    ]
    totwidth = sum(sum.(A)) + 2 * (length(A) - 1)

    #intert my stuff which requires totwidth
    avgaggcoef = round(mean(m.agg_coef); sigdigits = 3)
    ctitle = string(typeof(m)) * " (Aggregate coef: $avgaggcoef)"
    halfwidth = div(totwidth - length(ctitle), 2)
    println(io, " "^halfwidth * ctitle * " "^halfwidth)
    return show(io, coeftable(m))
end

struct ObservationIndex
    start_indices::Vector{Int}  # Starting index for each time period
    end_indices::Vector{Int}    # Ending index for each time period
    ids::Vector{Int}            # Entity IDs for each observation
    entity_obs_indices::Matrix{Int}  # N×T matrix: entity_obs_indices[i,t] = observation index of entity i in period t, or 0 if not present
    N::Int                      # Total number of entities 
    T::Int                      # Total number of time periods
end

function create_observation_index(df, id, t)
    # Ensure data is sorted by (time, id)
    if !issorted(df, [t, id])
        sort!(df, [t, id])
    end

    N = length(unique(df[!, id]))
    T = length(unique(df[!, t]))

    # Get id mapping
    id_map = Dict(unique(df[!, id]) .=> 1:N)
    time_map = Dict(unique(df[!, t]) .=> 1:T)

    # Create vectors to store indices
    start_indices = zeros(Int, T)
    end_indices = zeros(Int, T)
    ids = [id_map[row[id]] for row in eachrow(df)]

    # Create matrix to store observation indices for each entity in each period
    # 0 indicates entity is not present in that period
    entity_obs_indices = zeros(Int, N, T)

    # Find start and end indices for each time period
    current_time = time_map[df[1, t]]
    start_indices[current_time] = 1

    for i in 2:nrow(df)
        time_idx = time_map[df[i, t]]
        if time_idx != current_time
            end_indices[current_time] = i - 1
            start_indices[time_idx] = i
            current_time = time_idx
        end
    end
    end_indices[current_time] = nrow(df)

    # Fill the entity_obs_indices matrix
    for i in 1:nrow(df)
        entity_id = id_map[df[i, id]]
        time_id = time_map[df[i, t]]
        entity_obs_indices[entity_id, time_id] = i  # Store actual observation index
    end

    return ObservationIndex(start_indices, end_indices, ids, entity_obs_indices, N, T)
end