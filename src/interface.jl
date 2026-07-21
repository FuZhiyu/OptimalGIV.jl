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
- `precision_mode::Union{Nothing,Symbol} = nothing`: Entity precision-weighting scheme.
    `nothing` (the default) resolves to `:twostep` — the package default changed from
    `:cue` in v0.3.0 — and emits a one-time (per session) warning; pass any mode
    explicitly, or set `quiet = true`, to silence it. (`:scalar_search` ignores
    precision weighting entirely and keeps its previous behavior silently.)
    - `:twostep` (package default): two-step efficient GMM. Step 1 solves with the
      `:proxy` weights; step 2 recomputes the precisions `1/var(ûᵢ)` from the step-1
      residuals and re-solves once with those fixed weights, warm-started at the step-1
      root. Estimates and SEs come from step 2; `converged` requires both steps (if
      step 1 fails, step 2 is skipped and the non-converged step-1 estimates are
      returned). The two-step is the standard efficient-GMM truncation: its fixed point
      under further iteration is exactly a CUE root (the system is exactly identified),
      but iterating is deliberately not pursued because it re-imports the CUE
      self-weighting instability.
    - `:cue`: continuously-updated GMM weights `1/σᵢ²(ζ)`, recomputed each solver
      evaluation (the previous default; results are byte-identical to it when pinned
      explicitly).
    - `:proxy`: fixed data-based precisions `1/var(uqᵢ)` from the FE/control-residualized
      flows, computed once before solving. Under incomplete coverage this makes the moment
      map an exact fixed quadratic in ζ, removing the CUE self-weighting instability.
    - `:fixed`: user-supplied `precision_weights` (`:twostep` ≡ `:proxy` followed by
      `:fixed` at the step-1-residual precisions).
- `precision_weights::Union{Nothing,AbstractVector} = nothing`: precision vector (length `N`,
    sorted entity order) used when `precision_mode = :fixed`.

    Standard errors: with `precision_mode = :cue` and complete coverage, the CUE-optimal
    vcov is used; in every other case SEs route through the general sandwich `solve_vcov`,
    carrying the same fixed precisions as the moments (for `:twostep`, the step-2
    precisions) and, under complete coverage, the same period `Mweights` (evaluated at
    the solution) that scaled the solved moments.
- `method::Symbol = :trust_region`, `autodiff::Symbol = :central`: passed through to
    NLsolve.jl. `autodiff = :forward` uses ForwardDiff for the Jacobian (the moment code is
    generic in `eltype(ζ)`). Defaults are NLsolve's own defaults, so default behavior is unchanged.
- `jacobian::Symbol = :analytic`: Jacobian supplied to the NLsolve solve. `:analytic`
    (default) uses the exact hand-coded Jacobian of the fixed-precision moment map whenever
    it applies — fixed precisions (`:proxy`/`:fixed` and both `:twostep` steps) with
    algorithm `:iv`/`:iv_twopass` and no internal PCs; in every other case (CUE, internal
    PCs, `:debiased_ols`) the solver falls back to the `autodiff` path automatically.
    `jacobian = :autodiff` forces the finite-difference/ForwardDiff Jacobian everywhere
    (debugging escape hatch).
- `pc_solver::Symbol = :onestep`: Solve strategy for internal-PC specs (`pc(k)` in the
    formula, `n_pcs > 0`). `:onestep` (default) re-extracts the PCs inside every moment
    evaluation and solves in one NLsolve pass. `:nested` runs a Bai-style iterated-PC
    outer loop (extract PCs at fixed ζ) around an inner fixed-quadratic ζ solve with the
    exact analytic Jacobian; applies only to fixed-precision `:iv`/`:iv_twopass` specs and
    is ignored otherwise. Both converge to the same joint fixed point; `:nested` exists as
    an alternative solve mechanism — see the nested-PC benchmark for the adoption call.
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


"""
    resolve_precision(precision_mode, precision_weights, uq, obs_index)

Resolve the entity precision-weight vector for the fixed-weight estimation modes.

- `:cue`  → `nothing`; the moment code keeps updating `1/σᵢ²(ζ)` each evaluation
  (continuously-updated GMM, the pre-v0.3.0 default).
- `:proxy` → `1/var(uqᵢ)` computed once from the FE/control-residualized flows,
  so the moment map is a fixed quadratic in ζ under incomplete coverage.
- `:twostep` → the step-1 `:proxy` weights. The second solve (precisions from the
  step-1 residuals) is orchestrated by `giv()`; `build_error_function` has no solve
  loop, so its exported error function is the step-1 (proxy-weighted) moment map.
- `:fixed` → the user-supplied `precision_weights` vector (length `N`, in sorted
  entity order).
"""
function resolve_precision(precision_mode, precision_weights, uq, obs_index)
    if precision_mode == :cue
        return nothing
    elseif precision_mode == :proxy || precision_mode == :twostep
        return 1 ./ calculate_entity_variance(uq, obs_index)
    elseif precision_mode == :fixed
        isnothing(precision_weights) &&
            throw(ArgumentError("precision_mode = :fixed requires a `precision_weights` vector (length N, sorted entity order)."))
        length(precision_weights) == obs_index.N ||
            throw(ArgumentError("`precision_weights` must have length N = $(obs_index.N) (got $(length(precision_weights)))."))
        return collect(float.(precision_weights))
    else
        throw(ArgumentError("Unknown precision_mode = $(precision_mode); use :cue, :proxy, :fixed, or :twostep."))
    end
end

# One-time (per session) announcement of the :cue → :twostep default change.
# The flag flips only when the warning is actually emitted, so `quiet = true`
# calls neither warn nor consume it.
const _TWOSTEP_DEFAULT_WARNED = Ref(false)

function default_precision_mode(quiet::Bool)
    if !quiet && !_TWOSTEP_DEFAULT_WARNED[]
        _TWOSTEP_DEFAULT_WARNED[] = true
        @warn "The default GIV estimator changed from :cue to :twostep (proxy first pass, " *
              "then one re-solve with precisions from the first-step residuals). Pin " *
              "`precision_mode = :cue` to reproduce previous results; pass any `precision_mode` " *
              "explicitly (or set `quiet = true`) to silence this one-time warning."
    end
    return :twostep
end

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
    precision_mode=nothing,  # sentinel: resolves to the package default :twostep (one-time warning)
    precision_weights=nothing,
    method=:trust_region,
    autodiff=:central,
    jacobian=:analytic,
    pc_solver=:onestep,
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

    # Resolve the `precision_mode = nothing` sentinel: the package default is
    # :twostep (since v0.3.0; previously :cue), announced by a one-time warning.
    # :scalar_search ignores precision weighting entirely, so its implicit
    # default keeps the previous (CUE-equivalent) behavior silently.
    if isnothing(precision_mode)
        precision_mode = algorithm == :scalar_search ? :cue : default_precision_mode(quiet)
    end

    # Fixed proxy/user precision weights (non-CUE weighting; :twostep resolves to
    # the step-1 proxy weights here). `nothing` keeps the continuously-updated
    # CUE weights.
    precisionvec = resolve_precision(precision_mode, precision_weights, uq, obs_index)

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
        precision=precisionvec,
        method=method,
        autodiff=autodiff,
        jacobian=jacobian,
        pc_solver=pc_solver,
    )

    # Two-step efficient GMM (:twostep, the package default): the solve above is
    # step 1 (:proxy weights); step 2 recomputes the precisions 1/var(ûᵢ) from the
    # step-1 residuals û₁ = uq + uCp ζ̂₁ and re-solves once with those fixed
    # weights, warm-started at ζ̂₁. Reported estimates and SEs come from step 2;
    # `converged` requires both steps. Iterating further would converge to a CUE
    # root (the system is exactly identified) — deliberately not pursued, as
    # iteration re-imports the CUE self-weighting instability.
    if precision_mode == :twostep && algorithm != :scalar_search
        if converged
            precisionvec = 1 ./ calculate_entity_variance(uq + uCp * ζ̂, obs_index)
            ζ̂, converged2 = estimate_giv(
                uq,
                uCp,
                C,
                S,
                obs_index,
                Val{algorithm}();
                guess=ζ̂,
                quiet=quiet,
                complete_coverage=complete_coverage,
                solver_options=solver_options,
                n_pcs=n_pcs,
                pca_option=pca_option,
                precision=precisionvec,
                method=method,
                autodiff=autodiff,
                jacobian=jacobian,
                pc_solver=pc_solver,
            )
            converged = converged && converged2
        elseif !quiet
            @warn "Two-step step 1 (:proxy) did not converge; skipping step 2 and returning the non-converged step-1 estimates."
        end
    end
    β_q = β_ols[:, 1]
    β_Cp = β_ols[:, 2:end]
    β = β_q + β_Cp * ζ̂

    û = uq + uCp * ζ̂
    if return_vcov && n_pcs == 0 # with internal PCs, the vcov calculation is off.
        # Vcov routing rule: the sandwich `solve_vcov` is the default everywhere;
        # `solve_optimal_vcov` only when the weights are exact CUE
        # (`precision_mode = :cue`) AND coverage is complete.
        if isnothing(precisionvec)
            if complete_coverage
                σu²vec, Σζ = solve_optimal_vcov(ζ̂, û, S, C, obs_index)
            else
                # without complete coverage of the market, we do not have aggregate elasticity
                # and hence it's not exactly optimal
                σu²vec, Σζ = solve_vcov(û, S, C, uCp, obs_index)
            end
        else
            # Fixed-weight mode: SEs use the same fixed weights as the moments
            # (not the CUE-optimal weights), via the general sandwich. When the
            # `:iv` kernels solved period-Mweighted moments (complete coverage),
            # carry the same Mweights evaluated at the solution into the sandwich `W`.
            Mw = complete_coverage && algorithm in (:iv, :iv_twopass) ?
                 period_mweights(ζ̂, C, S, obs_index) : nothing
            σu²vec, Σζ = solve_vcov(û, S, C, uCp, obs_index; precision=precisionvec, Mweights=Mw)
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

`precision_mode = nothing` (the default) resolves to the package default `:twostep`,
which here means the **step-1 proxy weights** `1/var(uqᵢ)` — the export has no solve
loop, so the returned function is the step-1 (fixed-quadratic) moment map. Pass
`precision_mode = :cue` for the continuously-updated map, or `:fixed` with
step-1-residual precisions for the step-2 map.

When the exported map has an exact closed-form Jacobian (fixed precisions,
algorithm `:iv`/`:iv_twopass`, no internal PCs), the returned NamedTuple carries it
as `jac_func` (`ζ -> Nmom×Nmom` matrix, via [`mean_moment_jacobian`](@ref)); it is
`nothing` otherwise. Useful for root diagnostics (SVD/weak-direction analysis at ζ̂)
without finite-differencing the moment map.
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
    precision_mode=nothing,  # sentinel: package default :twostep ⇒ step-1 proxy weights here
    precision_weights=nothing,
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
        # `nothing` sentinel → package default :twostep, which resolves to the
        # step-1 proxy weights here (no solve loop in this export).
        if isnothing(precision_mode)
            precision_mode = default_precision_mode(quiet)
        end
        precisionvec = resolve_precision(precision_mode, precision_weights, uq, obs_index)
        err_func = x -> mean_moment_conditions(x, uq, uCp, C, S, obs_index, complete_coverage, Val{algorithm}(), n_pcs, pca_option; precision=precisionvec)
        # Exact Jacobian of the exported moment map, when defined (fixed precisions,
        # `:iv`/`:iv_twopass`, no internal PCs); `nothing` otherwise.
        jac_func = (!isnothing(precisionvec) && n_pcs == 0 && algorithm in (:iv, :iv_twopass)) ?
                   (x -> mean_moment_jacobian(x, uq, uCp, C, S, obs_index, complete_coverage, precisionvec)) :
                   nothing
        return err_func, (uq=uq, uCp=uCp, C=C, S=S, obs_index=obs_index, n_pcs=n_pcs, precision=precisionvec, jac_func=jac_func)
    end
end