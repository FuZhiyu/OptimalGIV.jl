"""
    giv(df, formula, id, t, weight; <keyword arguments>)


Estimate the GIV model given by:

```math
    q_it +  endog(p_t) × C_it' ζ = X_it' β + u_it
```
such that p_t is pinned down by market clearing condition Σ_i (q_it S_it) = 0, and E[u_it u_jt] = 0. 

It returns a `GIVModel` object containing the estimated coefficients, standard errors, and other information.

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
    ```
- `exclude_pairs::Dict{Int,Vector{Int}} = Dict()`: A dictionary specifying entity pairs to exclude from the moment conditions. 
    Keys are entity IDs and values are vectors of entity IDs to exclude. For example:
    ```julia
    exclude_pairs = Dict(1 => [2, 3], 4 => [5])  # Exclude pairs (1,2), (1,3), (4,5)
    ```
- `algorithm::Symbol = :iv`: The algorithm to use for estimation. The default is `:iv`. The options are
    - `:iv`: The most flexible algorithm. It uses the moment condition such that E[u_i u_{S,-i}] = 0. 
    This algorithm uses an identity to achieve O(N) computational complexity.
    - `:iv_twopass`: Numerically identical to `:iv` but uses a more straightforward O(N²) implementation. 
    Useful for debugging or when the O(N) trick causes numerical issues.
    - `:debiased_ols`: `:debiased_ols` uses the moment condition such that E[u_i C_it p_it] = 1/ζ_St σ_i^2. ]
    It requires the adding-up constraint is satisifed so that Σ_i (q_it weight_i) = 0. 
    If not, the aggregate elasticity will be underestimated.
    - `:scalar_search`: `:scalar_search` uses the same moment condition `up` but requires the aggregate elasticity be constant across time. 
    It searches for the scalar of the aggregate elasticity and hence very efficient. 
    It can be used for diagnoises or forming initial guess for other algorithms. 
- `quiet::Bool = false`: If `true`, suppress warnings and information messages.
- `save::Symbol = :none`: Controls what additional information to save:
    - `:none`: Save only the coefficients and standard errors (default)
    - `:residuals`: Save residuals in the returned model
    - `:fe`: Save fixed effects estimates
    - `:all`: Save both residuals and fixed effects
- `save_df::Bool = false`: If `true`, the processed estimation DataFrame (including residuals, fixed-effects, and coefficient columns when requested) is stored in the returned model under `df`. This can be useful for post-estimation analysis but increases memory usage.
- `complete_coverage::Union{Nothing,Bool} = nothing`: Whether entities cover the full market. 
    If `nothing` (default), automatically detected by checking the market clearing condition. 
    Can be manually set to `true` or `false` for debugging purposes.
- `return_vcov::Bool = true`: Whether to calculate and return the variance-covariance matrix.
- `contrasts::Dict{Symbol,Any} = Dict()`: Contrasts specification for categorical variables (following StatsModels.jl conventions). Untested. Use with caution.
- `tol::Float64 = 1e-6`: Convergence tolerance for the solver and fixed effects.
- `iterations::Int = 100`: Maximum number of iterations for the solver.
- `solver_options::NamedTuple`: Additional options to pass to NLsolve.jl. 
    Default is `(; ftol=tol, show_trace=!quiet, iterations=iterations)`.
- `pca_option::NamedTuple`: Additional options to pass to HeteroPCA.heteropca(). 
    Default is `(; impute_method=:zero, demean=false, maxiter=1000, algorithm=DeflatedHeteroPCA(t_block=10))`.

# Output

The output is `m::GIVModel`. Several important fields are:

  - `endog_coef`: Coefficients on endogenous terms (vector `ζ̂`).
  - `exog_coef`: Coefficients on exogenous control variables (vector `β`).
  - `endog_vcov`: Variance-covariance matrix of `endog_coef`.
  - `exog_vcov`: Variance-covariance matrix of `exog_coef`.
  - `agg_coef`: Aggregate (or average) elasticity. Scalar if constant across time, otherwise a vector indexed by `t`.
  - `residual_variance`: Estimated residual variance for each `id`.
  - `coefdf::DataFrame`: Tidy DataFrame with entity-specific coefficients (and, when requested, fixed effects).
  - `df::Union{DataFrame,Nothing}`: If `save_df = true`, the processed estimation dataset augmented with residuals, coefficients, and fixed-effects columns.

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
    save_df=false,
    complete_coverage=nothing, # if nothing, we check the market clearing to determine. You can overwrite it using this keyword. 
    return_vcov=true,
    contrasts=Dict{Symbol,Any}(), # not tested; 
    tol=1e-6,
    iterations=100,
    solver_options=(; ftol=tol, show_trace=!quiet, iterations=iterations),
    pca_option=(; impute_method=:zero, demean=false, maxiter=100, algorithm=DeflatedHeteroPCA(t_block=10)),
)
    formula = replace_function_term(formula) # FunctionTerm is inconvenient for saving&loading across Module
    df = preprocess_dataframe(df, formula, id, t, weight)
    formula_givcore, formula_schema, fes, feids, fekeys, n_pcs = separate_giv_ols_fe_formulas(df, formula; contrasts=contrasts)
    # regress the left-hand side q, and Cp on the right-hand side

    response_name, endog_name, endog_coefnames, exog_coefnames, slope_terms = get_coefnames(formula_givcore, formula_schema)

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
        throw(ArgumentError("Without complete coverage of the whole market, `up` and `scalar_search` algorithms should not be used. You can overwrite it by forcing the keyword `complete_coverage` to `true`."))
    end

    guessvec = parse_guess(endog_coefnames, guess, Val{algorithm}())
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
        n_pcs=n_pcs,
        pca_option=pca_option,
    )
    β_q = β_ols[:, 1]
    β_Cp = β_ols[:, 2:end]
    β = β_q + β_Cp * ζ̂

    û = uq + uCp * ζ̂
    if return_vcov && n_pcs == 0 # with internal PCs, the vcov calculation is off. 
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
    coefdf = create_coef_dataframe(df, formula_schema, coef, id; fekeys=fekeys)
    if (save == :all || save == :fe) && length(feids) > 0
        fedf = select(df, fekeys)
        fedf = retrieve_fixedeffects!(fedf, Y_original * [one(eltype(ζ̂)); ζ̂] - X_original * β, feM, feids)
        unique!(fedf, fekeys)
        # Merge fixed effects into coefdf as documented
        coefdf = leftjoin(coefdf, fedf, on=intersect(names(coefdf), names(fedf)))
    else
        fedf = nothing
    end

    ζS = solve_aggregate_elasticity(ζ̂, C, S, obs_index; complete_coverage=complete_coverage)
    ζS = length(unique(ζS)) == 1 ? ζS[1] : ζS

    # Extract PCs from final residuals if requested and update residuals
    pc_factors = nothing
    pc_loadings = nothing
    pc_model = nothing

    if n_pcs > 0
        pc_factors, _, pc_model, û = extract_pcs_from_residuals(û, obs_index, n_pcs; pca_option...)
        # !the saved û is the residuals after the PCs
        pc_loadings = projection(pc_model) # important: save the projection so that projection x factors = predicted values; loading(pc_model) will include the factor vol as well. 
    end

    dof = length(ζ̂) + length(β)
    dof_residual = nrow(df) - dof

    if save == :residuals || save == :all
        resdf = select(df, id, t)
        resdf[!, Symbol(response_name, "_residual")] = û
    else
        resdf = nothing
    end

    if save_df
        savedf = df
        if !isnothing(resdf)
            savedf[!, Symbol(response_name, "_residual")] = û
        end
        # Join coefficient DataFrame with main DataFrame
        common_cols = intersect(names(savedf), names(coefdf))
        if !isempty(common_cols)
            # Normal case: join on categorical variables
            savedf = leftjoin(savedf, coefdf, on=common_cols)
        elseif nrow(coefdf) == 1
            # No categorical variables: broadcast single row of coefficients to all rows
            # This happens when all terms are continuous variables
            savedf = crossjoin(savedf, coefdf)
        else
            # This should never happen - multiple rows but no common columns
            throw(ArgumentError("Coefficient DataFrame has multiple rows but no columns in common with data. This indicates a bug in create_coef_dataframe."))
        end

        if !isnothing(fedf)
            savedf = leftjoin(savedf, fedf, on=intersect(names(savedf), names(fedf)))
        end
        sort!(savedf, [t, id])
    else
        savedf = nothing
    end


    # If saving dataframe and PC factors were extracted, add them to savedf
    if save_df && n_pcs > 0 && !isnothing(pc_factors)

        # Add PC factors to savedf (factors are k×T, so we need pc_factors[k, :])
        # IMPORTANT: Use sorted time order to match ObservationIndex ordering
        # ObservationIndex sorts time periods, so pc_factors columns correspond to sorted times
        time_pc_df = DataFrame(t => sort(unique(df[!, t])))
        for k in 1:n_pcs
            time_pc_df[!, Symbol("pc_factor_", k)] = pc_factors[k, :]
        end
        savedf = leftjoin(savedf, time_pc_df, on=t)

        # Add PC loadings to savedf (loadings are by entity)
        if !isnothing(pc_loadings)
            # CRITICAL: Use sorted entity order to match ObservationIndex ordering
            # ObservationIndex uses natural sort order for entities (line 66 of observation_index.jl)
            # This ensures PC loadings are correctly aligned with entity indices
            entity_loading_df = DataFrame(id => sort(unique(df[!, id])))
            for k in 1:n_pcs
                entity_loading_df[!, Symbol("pc_loading_", k)] = pc_loadings[:, k]
            end
            savedf = leftjoin(savedf, entity_loading_df, on=id)
        end
    end

    return GIVModel(
        ζ̂,
        β,
        Σζ,
        Σβ,
        σu²vec,
        ζS,
        complete_coverage,

        formula,
        formula_schema,
        response_name,
        endog_name,
        endog_coefnames,
        exog_coefnames,

        id,
        t,
        weight,
        Dict(exclude_pairs),

        coefdf,
        fedf,
        resdf,
        savedf,

        # PC-related fields
        n_pcs,
        pc_factors,
        pc_loadings,
        pc_model,

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
    formula_givcore, formula_schema, fes, feids, fekeys, n_pcs = separate_giv_ols_fe_formulas(df, formula; contrasts=contrasts)
    return get_coefnames(formula_givcore, formula_schema)
end

function get_coefnames(formula_givcore, formula_schema)
    slope_terms, endog_term = parse_endog(formula_givcore)
    endog_name = string(endogsymbol(endog_term))

    response_name = coefnames(formula_schema.lhs)[1]
    endog_coefnames = coefnames(formula_schema.lhs)[2:end]
    exog_coefnames = coefnames(formula_schema.rhs)

    return response_name, endog_name, endog_coefnames, exog_coefnames, slope_terms
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

parse_guess(endog_coefnames, guess::Vector{<:Number}, ::Val) = guess
parse_guess(endog_coefnames, ::Nothing, ::Val) = nothing
parse_guess(endog_coefnames, guess::Number, ::Val) = [guess]

function parse_guess(endog_coefnames, guess::Dict, ::Val{:scalar_search})
    if "Aggregate" ∉ keys(guess)
        throw(ArgumentError("To use the scalar-search algorithm, specify the initial guess using \"Aggregate\" as the key in `guess`"))
    end
    return parse_guess(endog_coefnames, guess["Aggregate"], Val(:scalar_search))
end

function parse_guess(endog_coefnames, guess::Dict, ::Any)
    guess = Dict(string(k) => v for (k, v) in pairs(guess))
    return map(endog_coefnames) do endog_coefname
        if endog_coefname ∉ keys(guess)
            throw(ArgumentError("Initial guess for \"$(endog_coefname)\" is missing"))
        else
            return guess[endog_coefname]
        end
    end
end

"""
    extract_raw_matrices(df, formula, id, t, weight)
    
    Convenient function to extract the raw matrices used for estimation. 
"""
function extract_raw_matrices(df, formula, id, t, weight; contrasts=Dict{Symbol,Any}(), exclude_pairs=Dict{Int,Vector{Int}}())
    formula = replace_function_term(formula) # FunctionTerm is inconvenient for saving&loading across Module
    df = preprocess_dataframe(df, formula, id, t, weight)
    formula_givcore, formula_schema, fes, feids, fekeys, n_pcs = separate_giv_ols_fe_formulas(df, formula; contrasts=contrasts)
    # regress the left-hand side q, and Cp on the right-hand side

    response_name, endog_name, endog_coefnames, exog_coefnames, slope_terms = get_coefnames(formula_givcore, formula_schema)

    X_original = convert(Matrix{Float64}, modelcols(formula_schema.rhs, df))
    Y_original = modelcols(collect_matrix_terms(formula_schema.lhs), df)
    q, Cp = Y_original[:, 1], Y_original[:, 2:end]
    # Y_feres, X_feres, β_ols, residuals, feM = ols_with_fixed_effects(Y_original, X_original, fes; tol=tol)

    # uq, uCp = residuals[:, 1], residuals[:, 2:end]

    S = df[!, weight]
    obs_index = create_observation_index(df, id, t, exclude_pairs)

    formula_slope = apply_schema(slope_terms, FullRank(schema(slope_terms, df, contrasts)))
    C = modelcols(collect_matrix_terms(formula_slope), df)
    return q, Cp, C, S, X_original, obs_index
end

function create_coef_dataframe(df, formula_schema, coef, id; fekeys=[])
    slope_terms = eachterm(formula_schema.lhs[2:end])
    exog_terms = eachterm(formula_schema.rhs.terms)
    terms = [slope_terms..., exog_terms...]
    categorical_terms_symbol = Symbol[]
    cat_symbol(t::CategoricalTerm) = [Symbol(t)]
    cat_symbol(t::InteractionTerm) = [Symbol(x) for x in t.terms if x isa CategoricalTerm]
    categorical_terms_symbol = [cat_symbol(t) for t in terms if has_categorical(t)]
    categorical_terms_symbol = unique(vcat(categorical_terms_symbol..., fekeys))
    if length(categorical_terms_symbol) == 0 
        # No categorical terms: create single-row DataFrame for coefficients
        # Use a temporary placeholder column that will be removed later
        categories = DataFrame(:_placeholder_ => [1])
    else
        categories = select(df, categorical_terms_symbol) |> unique
    end

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
            categories[!, coefsym] = catmat * termcoef
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
    if length(categorical_terms_symbol) == 0
        # Remove the placeholder column, leaving only coefficient columns
        select!(categories, Not(:_placeholder_))
    end

    return categories
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
        if weighted_sum^2 > sqrt(eps(eltype(q)))
            return false  # Constraint violated
        end
    end

    return true  # No violations found
end

"""
    build_error_function(df, formula, id, t, weight; <keyword arguments>)

Export the error function for the GIV model. This function is useful for debugging and customized solvers. 
"""
function build_error_function(df,
    formula::FormulaTerm,
    id::Symbol,
    t::Symbol,
    weight::Union{Symbol,Nothing}=nothing;
    exclude_pairs=Dict{Int,Vector{Int}}(),
    algorithm=:iv,
    quiet=false,
    complete_coverage=nothing, # if nothing, we check the market clearing to determine. You can overwrite it using this keyword. 
    contrasts=Dict{Symbol,Any}(), # not tested; 
    tol=1e-6,
    pca_option=(; impute_method=:zero, demean=false, maxiter=1000),
    kwargs...
)
    formula = replace_function_term(formula) # FunctionTerm is inconvenient for saving&loading across Module
    df = preprocess_dataframe(df, formula, id, t, weight)
    formula_givcore, formula_schema, fes, feids, fekeys, n_pcs = separate_giv_ols_fe_formulas(df, formula; contrasts=contrasts)
    # regress the left-hand side q, and Cp on the right-hand side

    response_name, endog_name, endog_coefnames, exog_coefnames, slope_terms = get_coefnames(formula_givcore, formula_schema)

    X_original = convert(Matrix{Float64}, modelcols(formula_schema.rhs, df))
    Y_original = modelcols(collect_matrix_terms(formula_schema.lhs), df)
    q, Cp = Y_original[:, 1], Y_original[:, 2:end]
    Y_feres, X_feres, β_ols, residuals, feM = ols_with_fixed_effects(Y_original, X_original, fes; tol=tol)

    uq, uCp = residuals[:, 1], residuals[:, 2:end]

    S = df[!, weight]
    obs_index = create_observation_index(df, id, t, exclude_pairs)

    formula_slope = apply_schema(slope_terms, FullRank(schema(slope_terms, df, contrasts)))
    C = modelcols(collect_matrix_terms(formula_slope), df)

    if isnothing(complete_coverage)
        complete_coverage = check_market_clearing(q, S, obs_index)
    end
    if !quiet &&
       algorithm ∈ [:scalar_search, :debiased_ols] &&
       !complete_coverage
        throw(ArgumentError("Without complete coverage of the whole market, `up` and `scalar_search` algorithms should not be used. You can overwrite it by forcing the keyword `complete_coverage` to `true`."))
    end
    if algorithm == :scalar_search
        # Check if panel is balanced before proceeding
        N, T = obs_index.N, obs_index.T

        # Verify that we have a balanced panel (N*T observations)
        if length(q) != N * T
            throw(ArgumentError("Scalar search algorithm requires a balanced panel"))
        end

        # Reshape stacked vectors back to matrices/tensors
        uqmat = reshape(uq, N, T)

        # For C, reshape from (N*T, K) to (N, T, K)
        Nmom = size(C, 2)
        Cts = BitArray(reshape(C, N, T, Nmom))
        uCpts = reshape(uCp, N, T, Nmom)

        Smat = reshape(S, N, T)

        # Call the original implementation with reshaped matrices
        p, S_vec, coefmapping = transform_matricies_for_scalar_search(uCpts, Cts, Smat)

        err_func = x -> ζS_err(x, uqmat, p, S_vec, coefmapping; kwargs...)
        return err_func, (uqmat=uqmat, p=p, S_vec=S_vec, coefmapping=coefmapping)
    else
        err_func = x -> mean_moment_conditions(x, uq, uCp, C, S, obs_index, complete_coverage, Val{algorithm}(), n_pcs, pca_option)
        return err_func, (uq=uq, uCp=uCp, C=C, S=S, obs_index=obs_index, n_pcs=n_pcs)
    end
end