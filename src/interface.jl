"""
    giv(df, formula, id, t, weight; <keyword arguments>)


Estimate the GIV model. It returns a `GIVModel` object containing the estimated coefficients, standard errors, and other information.


# Arguments

- `df::DataFrame`: A DataFrame containing the data. Only balanced panel is supported for now. 
    It is recommended to sort the data by `t` and `id`.
- `formula::FormulaTerm`: A formula specifying the model. The formula should be in the form of
    `q + (C1 + C2+...) & endog(p) ~ exog_controls`, where 
        
    - `q` is the response variable, 
    - `endog(p)` indicates p is the endogenous variable,
    - `C1, C2, ...` can be the categorical variables to specify heterogeneous loadings, or exogenous variables to be interacted with the endogenous variable; when `x` are ommited, different entities are assumed to have the same loadings.
    - `exog_controls` are the exogenous variables. Notice that by default the model does not include an intercept term. If the mean is not zero, it is recommended to an entity fixed effect to demean the data.

    For example, `formula` can be written as
    ```julia
    @formula(q + id & endog(p) + C & endog(p) ~ id & η + id)
    ```


    Also notice that 
    - Endogenous variables are assumed to be on the left-hand side of the formula; 
    - All categorical&Bool variables are treated as fixed effects. 
- `id::Symbol`: The column name of the entity identifier.
- `t::Symbol`: The column name of the time identifier. `t` and `id` should uniquely identify each observation.
- `weight::Union{Symbol,Nothing}`: The column name of the weight variable for each entities. The weight must be non-negative. 
    You can flip swap the sign of `q` and `weight` if necessary. 

## Keyword Arguments

- `guess`: Initial guess for the coefficients in front of endogenous terms. If not provided, the initial guess is set using OLS. 
    Guess can be supplied in multiple ways:
    - A vector in the order of coefficient enters the formula. For categorical variables the order is determined by the variable.
    - A dictionary with the key being the name (either a string or a symbol) of the interaction term and the value being the initial guess 
    (a vector in the case of categorical variables and a number otherwise). In the example above, the initial guess can be provided as
    ```julia
    guess = Dict(:id => [1.0, 2.0], :η => 0.5)
- `algorithm::Symbol = :iv`: The algorithm to use for estimation. The default is `:iv`. The options are
    - `:iv`: The most flexible algorithm. It uses the moment condition such that E[u_i u_{S,-i}] = 0
    - `:debiased_ols`: `:debiased_ols` uses the moment condition such that E[u_i C_it p_it] = 1/ζ_St σ_i^2. ]
    It requires the adding-up constraint is satisifed so that Σ_i (q_it weight_i) = 0. 
    If not, the aggregate elasticity will be underestimated.
    - `:scalar_search`: `:scalar_search` uses the same moment condition `up` but requires the aggregate elasticity be constant across time. 
    It searches for the scalar of the aggregate elasticity and hence very efficient. 
    It can be used for diagnoises or forming initial guess for other algorithms. 
- `solver_options`: Options to pass to the `nlsolve`. 
- `quiet::Bool = false`: If `true`, suppress warnings and information.

  - `savedf::Bool = true`: If `true`, the input dataframe is saved in the field `df` of the returned `GIVModel` object. By default, it is `true`. For large datasets or repeated estimation, it is recommended to set it to `false`.


# Output

The output is `m::GIVModel`. Several important fields are:

  - `coef`: The estimated coefficients in front of the endogenous terms.
  - `vcov`: The estimated covariance matrix of the `coef`.
  - `factor_coef`: The estimated factor coefficients.
  - `agg_coef`: The estimated aggregate elasticity by `t`. If it is constant across `t`, it is stored as a scalar.
  - `residual_variance`: The estimated variance of the residual for each `id`.
  - `coefdf::DataFrame`: A `DataFrame`` containing the estimated coefficients.
  - `df::DataFrame`: A dataframe contains data used for estimation and the estimates.

"""
function giv(
    df,
    formula::FormulaTerm,
    id::Symbol,
    t::Symbol,
    weight::Union{Symbol,Nothing}=nothing;
    guess=nothing,
    exclude_pairs=Dict{Int,Vector{Int}}(),
    algorithm=:iv,
    quiet=false,
    save=:none, # :all or :fe or :none or :residuals
    complete_coverage=nothing, # by default we assume we observe the universe
    return_vcov=true,
    contrasts=Dict{Symbol,Any}(), # not tested; 
    tol=1e-6,
    iterations=100,
    solver_options=(; ftol=tol, show_trace=!quiet, iterations=iterations),
)
    formula = replace_function_term(formula) # FunctionTerm is inconvenient for saving&loading across Module
    df = preprocess_dataframe(df, formula, id, t, weight)
    formula_givcore, formula_schema, fes, feids, fekeys = separate_giv_ols_fe_formulas(df, formula; contrasts=contrasts)
    # regress the left-hand side q, and Cp on the right-hand side

    response_name, endog_name, elasticity_names, covariates_names, slope_terms = get_coefnames(formula_givcore, formula_schema)

    X_original = convert(Matrix{Float64}, modelcols(formula_schema.rhs, df))
    Y_original = modelcols(collect_matrix_terms(formula_schema.lhs), df)
    q, Cp = Y_original[:, 1], Y_original[:, 2:end]
    Y_feres, X_feres, β_ols, residuals, feM = ols_with_fixed_effects(Y_original, X_original, fes; tol=tol)

    uq, uCp = residuals[:, 1], residuals[:, 2:end]

    S = df[!, weight]
    obs_index = create_observation_index(df, id, t, exclude_pairs)

    formula_slope = apply_schema(slope_terms, FullRank(schema(slope_terms, df, contrasts)))
    C = modelcols(collect_matrix_terms(formula_slope), df)
    @assert size(C, 2) == size(Cp, 2)

    N = obs_index.N
    Nζ = size(C, 2)
    Nβ = size(X_original, 2)

    if isnothing(complete_coverage)
        complete_coverage = check_market_clearing(q, S, obs_index)
    end
    if !quiet &&
       algorithm ∈ [:scalar_search, :debiased_ols] &&
       !complete_coverage
        @error("Market clearing condition not satisfied. `up` and `scalar_search` algorithms may be biased.")
    end

    guessvec = parse_guess(formula, guess, Val{algorithm}())
    ζ̂, converged = estimate_giv(
        uq,
        uCp,
        C,
        S,
        obs_index,
        Val{algorithm}();
        guess=guessvec,
        quiet=quiet,
        complete_coverage=complete_coverage,
        solver_options=solver_options,
    )
    β_q = β_ols[:, 1]
    β_Cp = β_ols[:, 2:end]
    β = β_q + β_Cp * ζ̂

    û = uq + uCp * ζ̂
    if return_vcov
        if complete_coverage
            σu²vec, Σζ = solve_optimal_vcov(ζ̂, û, S, C, obs_index)
        else
            # without complete coverage of the market, we do not have aggregate elasticity
            # and hence it's not exactly optimal
            σu²vec, Σζ = solve_vcov(û, S, C, uCp, obs_index)
        end
        if size(X_feres, 2) > 0
            ols_vcov = solve_ols_vcov(σu²vec, X_feres, obs_index)
            Σβ = ols_vcov + β_Cp * Σζ * β_Cp'
        else
            Σβ = zeros(0, 0)
        end
    else
        σu²vec, Σζ, Σβ = NaN * zeros(N), NaN * zeros(Nζ, Nζ), NaN * zeros(Nβ, Nβ)
    end
    coef = [ζ̂; β]
    coef_names = [elasticity_names; covariates_names]
    vcov = [Σζ fill(NaN, Nζ, Nβ); fill(NaN, Nβ, Nζ) Σβ]
    coefdf = create_coef_dataframe(df, formula_schema, coef, id)
    if (save == :all || save == :fe) && length(feids) > 0
        fedf = select(df, fekeys)
        fedf = retrieve_fixedeffects!(fedf, Y_original * [one(eltype(ζ̂)); ζ̂] - X_original * β, feM, feids)
        unique!(fedf, fekeys)
        coefdf = innerjoin(coefdf, fedf; on=intersect(names(coefdf), names(fedf)))
    end


    ζS = solve_aggregate_elasticity(ζ̂, C, S, obs_index; complete_coverage=complete_coverage)
    ζS = length(unique(ζS)) == 1 ? ζS[1] : ζS

    dof = length(ζ̂) + length(β)
    dof_residual = nrow(df) - dof

    if save == :residuals || save == :all
        resdf = select(df, id, t)
        resdf[!, Symbol(response_name, "_residual")] = û
        resdf = innerjoin(resdf, coefdf; on=intersect(names(resdf), names(coefdf)))
        sort!(resdf, [t, id])
    else
        resdf = nothing
    end
    return GIVModel(
        coef,
        vcov,
        Nζ,
        Nβ,
        σu²vec,
        ζS,
        complete_coverage,

        formula,
        response_name,
        endog_name,
        coef_names,

        id,
        t,
        weight,
        exclude_pairs,

        coefdf,
        resdf,

        converged,
        N,
        obs_index.T,
        nrow(df),
        dof,
        dof_residual,
    )
end

"""
    get_coefnames(df::DataFrame, formula; contrasts=Dict{Symbol,Any}())

A convenience function to obtain the name of variables from the original formula. 
"""
function get_coefnames(df::DataFrame, formula; contrasts=Dict{Symbol,Any}())
    formula_givcore, formula_schema, fes, feids, fekeys = separate_giv_ols_fe_formulas(df, formula; contrasts=contrasts)
    return get_coefnames(formula_givcore, formula_schema)
end

function get_coefnames(formula_givcore, formula_schema)
    slope_terms, endog_term = parse_endog(formula_givcore)
    endog_name = string(endogsymbol(endog_term))

    response_name = coefnames(formula_schema.lhs)[1]
    elasticity_names = coefnames(formula_schema.lhs)[2:end]
    covariates_names = coefnames(formula_schema.rhs)

    return response_name, endog_name, elasticity_names, covariates_names, slope_terms
end


function retrieve_fixedeffects!(fedf, u, feM, feids; tol=1e-6, maxiter=100)
    newfes, b, c = solve_coefficients!(u, feM; tol=tol, maxiter=maxiter)
    for j in eachindex(newfes)
        fedf[!, feids[j]] = newfes[j]
    end
    return fedf
end

function preprocess_dataframe(df, formula, id, t, weight)
    # check data compatibility
    allvars = StatsModels.termvars(formula)
    df = select(df, unique([id, t, weight, allvars...]))
    dropmissing!(df)
    if any(nonunique(df, [id, t]))
        throw(ArgumentError("Observations are not uniquely identified by `id` and `t`"))
    end
    # check if any id has less than 2 observations
    if any(x -> nrow(x) < 2, groupby(df, id))
        throw(ArgumentError("Some entities have less than 2 observations. Please remove them."))
    end

    all(df[!, weight] .>= 0) ||
        throw(ArgumentError("Weight must be non-negative. You can swap the sign of y and S if necessary."))
    sort!(df, [t, id])
    return df
end

parse_guess(elasticity_names, guess::Vector{<:Number}, ::Val) = guess
parse_guess(elasticity_names, ::Nothing, ::Val) = nothing
parse_guess(elasticity_names, guess::Number, ::Val) = [guess]

function parse_guess(elasticity_names, guess::Dict, ::Val{:scalar_search})
    if "Aggregate" ∉ keys(guess)
        throw(ArgumentError("To use the scalar-search algorithm, specify the initial guess using \"Aggregate\" as the key in `guess`"))
    end
    return parse_guess(elasticity_names, guess["Aggregate"], Val(:scalar_search))
end

function parse_guess(elasticity_names, guess::Dict, ::Any)
    guess = Dict(string(k) => v for (k, v) in pairs(guess))
    return map(elasticity_names) do elasticity_name
        if elasticity_name ∉ keys(guess)
            throw(ArgumentError("Initial guess for \"$(elasticity_name)\" is missing"))
        else
            return guess[elasticity_name]
        end
    end
end

function create_coef_dataframe(df, formula_schema, coef, id)
    slope_terms = eachterm(formula_schema.lhs[2:end])
    exog_terms = eachterm(formula_schema.rhs.terms)
    terms = [slope_terms..., exog_terms...]
    categorical_terms_symbol = Symbol[]
    cat_symbol(t::CategoricalTerm) = [Symbol(t)]
    cat_symbol(t::InteractionTerm) = [Symbol(x) for x in t.terms if x isa CategoricalTerm]
    categorical_terms_symbol = [cat_symbol(t) for t in terms if has_categorical(t)]
    categorical_terms_symbol = unique(vcat(categorical_terms_symbol...))
    if categorical_terms_symbol == []
        categorical_terms_symbol = [id]
    end

    categories = select(df, categorical_terms_symbol) |> unique
    i = 1
    for term in terms
        termsym = Symbol(term)
        coefsym = Symbol(termsym, :_coef)
        # coefnamecol = Symbol(termsym, :_coefname)
        if has_categorical(term)
            if term isa InteractionTerm
                term = InteractionTerm((x for x in term.terms if x isa CategoricalTerm))
            end
            catmat = modelcols(term, categories)
            Nlevels = size(catmat)[2]
            termcoef = coef[i:i+Nlevels-1]
            termmat = modelcols(term, categories)
            categories[!, coefsym] = termmat * termcoef
            i += Nlevels
        elseif term isa InterceptTerm{false}
            continue
        else
            categories[!, coefsym] .= coef[i]
            i += 1
        end
    end
    if i != length(coef) + 1
        throw(ArgumentError("Number of coefficients does not match the number of terms. You may be using different formula or dataframe for estimation and creating coef dataframe."))
    end

    return categories
end

# """
#     predict_endog(m::GIVModel; <keyword arguments>)

# Predict the endogenous variable based on the estimated GIV model.

# # Arguments

# - `m::GIVModel`: The estimated GIV model.
# - `df::DataFrame`: The dataframe to predict the endogenous variable. 
#     It uses the saved dataframe in `m` by default. A different dataframe can be used for prediction to construct counterfactuals.

# ## Keyword Arguments
# - `coef::Vector{Float64}`: The estimated coefficients. 
#     It uses the coefficients in `m` by default. A different set of coefficients can be
#     used to construct counterfactuals.
# - `factor_coef::Vector{Float64}`: The estimated factor coefficients.
#     It uses the factor coefficients in `m` by default. A different set of factor coefficients can be
#     used to construct counterfactuals.
# - `residual::Symbol`: The name of the residual column in the dataframe. 
#     By default it uses the estimated residual; different residuals can be used to construct counterfactuals.
# - `formula::FormulaTerm`: The formula used to estimate the model.
# - `id::Symbol`: The name of the entity identifier.
# - `t::Symbol`: The name of the time identifier.
# - `weight::Symbol`: The name of the weight variable.
# """
# function predict_endog(
#     m::GIVModel,
#     df = m.df;
#     coef = m.coef,
#     factor_coef = m.factor_coef,
#     residual = Symbol(m.responsename, "_residual"),
#     formula = m.formula,
#     id = m.idvar,
#     t = m.tvar,
#     weight = m.weightvar,
#     quiet = false,
# )
#     if isnothing(df)
#         throw(ArgumentError("DataFrame not saved. Rerun the model with `savedf = true`"))
#     end
#     # add residual to the rhs
#     formula = FormulaTerm(formula.lhs, tuple(eachterm(formula.rhs)..., Term(residual)))
#     factor_coef = [factor_coef; one(eltype(factor_coef))]
#     df = preprocess_dataframe(df, formula, id, t, weight)
#     matricies = generate_matrices(df, formula, id, t, weight)
#     qmat, pmat, Cts, Cpts, ηts, Smat = matricies
#     N, T, Nmom = size(Cts)
#     mkc_err = sum(qmat .* Smat; dims = 1) .^ 2
#     if !quiet && any(>(eps(eltype(qmat))), mkc_err)
#         @warn ("Adding-up constraints not satisfied. Interpret the results with caution.")
#     end
#     aggcoef = solve_aggregate_elasticity(coef, Cts, Smat)
#     λη = reshape(ηts, N * T, :) * factor_coef
#     netshock = reshape(λη, N, T)
#     shockS = sum(netshock .* Smat; dims = 1) |> vec
#     pvec = shockS ./ aggcoef
#     return pvec
# end

function create_exclusion_matrix(id_values, exclude_pairs)
    # Create mapping from id to index
    id2ind = Dict(id_values[i] => i for i in 1:length(id_values))
    N = length(id_values)
    # Initialize exclusion matrix
    exclmat = zeros(Bool, N, N)

    # Fill in exclusions based on provided pairs
    for (id1, id2vec) in exclude_pairs
        i = id2ind[id1]
        for id2 in id2vec
            j = id2ind[id2]
            exclmat[i, j] = true
            exclmat[j, i] = true  # Make it symmetric
        end
    end

    return BitArray(exclmat)
end

"""
Check if the market clearing condition (adding-up constraint) is satisfied for each time period.

Returns true if the constraint is violated in any period.
"""
function check_market_clearing(q, S, obs_index)
    for t in 1:obs_index.T
        start_idx = obs_index.start_indices[t]
        end_idx = obs_index.end_indices[t]

        # Skip empty time periods
        if start_idx == 0 || end_idx == 0
            continue
        end

        # Calculate the sum of q*S for this time period
        weighted_sum = 0.0
        for idx in start_idx:end_idx
            weighted_sum += q[idx] * S[idx]
        end

        # Check if sum exceeds tolerance
        if weighted_sum^2 > eps(eltype(q))
            return false  # Constraint violated
        end
    end

    return true  # No violations found
end
