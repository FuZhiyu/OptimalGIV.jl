
"""
    resolve_precision(precision_weights, uq, obs_index)

Resolve the entity precision-weight vector for the fixed-weight estimation modes.

- `:cue`  → `nothing`; the moment code keeps updating `1/σᵢ²(ζ)` each evaluation
  (continuously-updated GMM, the pre-v0.3.0 default).
- `:raw_onestep` → `1/var(uqᵢ)` computed once from the FE/control-residualized flows,
  with equal period weights, so the IV moment map is a fixed quadratic in ζ under
  either coverage regime.
- `:twostep` → the step-1 `:raw_onestep` weights. `giv()` computes both applicable
  weight families at the first-step estimate and freezes them for the second solve;
  `build_error_function` has no solve loop, so its exported error function is the
  step-1 raw-weighted moment map.
- An entity-length vector is used directly as fixed precisions in sorted entity order.
"""
function resolve_precision(precision_weights, uq, obs_index)
    if precision_weights === :cue
        return nothing
    elseif precision_weights === :raw_onestep || precision_weights === :twostep
        return 1 ./ calculate_entity_variance(uq, obs_index)
    elseif precision_weights isa AbstractVector
        length(precision_weights) == obs_index.N ||
            throw(ArgumentError("`precision_weights` must have length N = $(obs_index.N) (got $(length(precision_weights)))."))
        return collect(float.(precision_weights))
    else
        throw(ArgumentError("Unknown precision_weights = $(precision_weights); use :twostep, :raw_onestep, :cue, or an entity-length vector."))
    end
end

# One-time migration warning for calls that omit `precision_weights`. The flag
# flips only when the warning is emitted, so quiet calls do not consume it.
const _TWOSTEP_DEFAULT_WARNED = Ref(false)

function default_precision_weights(quiet::Bool)
    if !quiet && !_TWOSTEP_DEFAULT_WARNED[]
        _TWOSTEP_DEFAULT_WARNED[] = true
        @warn "The default GIV estimator changed from :cue to :twostep. The two-step " *
              "estimator is more stable while retaining CUE-like efficiency, but estimates " *
              "and standard errors may differ slightly. " *
              "Pass `precision_weights = :twostep` explicitly (or set `quiet = true`) " *
              "to silence this one-time warning."
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
    complete_coverage::Bool,
    return_vcov=true,
    contrasts=Dict{Symbol,Any}(), # not tested;
    tol=1e-6,
    iterations=100,
    solver_options=(; ftol=tol, show_trace=!quiet, iterations=iterations),
    pca_option=(; impute_method=:zero, demean=false, maxiter=100, algorithm=DeflatedHeteroPCA(t_block=10)),
    precision_weights=nothing,  # omission sentinel; resolves to :twostep
    pin_zero=String[],
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

    if complete_coverage && !check_market_clearing(q, S, obs_index)
        throw(ArgumentError("`complete_coverage=true` requires market clearing in every period, but the supplied data fail the adding-up check."))
    end
    if algorithm ∈ [:scalar_search, :debiased_ols] && !complete_coverage
        throw(ArgumentError("`algorithm=$(algorithm)` requires `complete_coverage=true`."))
    end

    # Resolve omission to the two-step default. Scalar search ignores precision
    # weighting and therefore does not emit or consume the migration warning.
    if algorithm != :scalar_search && isnothing(precision_weights)
        precision_weights = default_precision_weights(quiet)
    end
    precisionvec = algorithm == :scalar_search ? nothing :
                   resolve_precision(precision_weights, uq, obs_index)

    # Pin selected endogenous coefficients at exactly zero and drop their moment
    # conditions. The reduced system is solved and expanded back to the full
    # coefficient vector after estimation.
    pin_idx = pin_zero_indices(pin_zero, endog_coefnames)
    if !isempty(pin_idx) && algorithm == :scalar_search
        throw(ArgumentError("`pin_zero` is not supported with `algorithm = :scalar_search`."))
    end
    keep_idx = setdiff(1:Nζ, pin_idx)
    C_free = isempty(pin_idx) ? C : C[:, keep_idx]
    uCp_free = isempty(pin_idx) ? uCp : uCp[:, keep_idx]

    guessvec = parse_guess(endog_coefnames, guess, Val{algorithm}())
    if !isempty(pin_idx) && guessvec isa AbstractVector
        length(guessvec) ∈ (Nζ, length(keep_idx)) || throw(ArgumentError(
            "with `pin_zero`, `guess` must have length $(Nζ) (full) or $(length(keep_idx)) (free coefficients only)."))
        guessvec = length(guessvec) == Nζ ? guessvec[keep_idx] : guessvec
    end
    Mweights = nothing
    ζ̂, converged = estimate_giv(
        uq,
        uCp_free,
        C_free,
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
    )

    # Two-step efficient GMM (:twostep, the package default): the solve above is
    # step 1 (:raw_onestep weights); step 2 recomputes the precisions 1/var(ûᵢ) from the
    # step-1 residuals û₁ = uq + uCp ζ̂₁ and re-solves once with those fixed
    # weights, warm-started at ζ̂₁. Reported estimates and SEs come from step 2;
    # `converged` requires both steps. Iterating further would converge to a CUE
    # root (the system is exactly identified) — deliberately not pursued, as
    # iteration re-imports the CUE self-weighting instability.
    if precision_weights === :twostep && algorithm != :scalar_search
        if converged
            precisionvec = 1 ./ calculate_entity_variance(uq + uCp_free * ζ̂, obs_index)
            Mweights = complete_coverage && algorithm in (:iv, :iv_twopass) ?
                       period_mweights(ζ̂, C_free, S, obs_index) : nothing
            ζ̂, converged2 = estimate_giv(
                uq,
                uCp_free,
                C_free,
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
                Mweights=Mweights,
            )
            converged = converged && converged2
        elseif !quiet
            @warn "Two-step step 1 (:raw_onestep) did not converge; skipping step 2 and returning the non-converged step-1 estimates."
        end
    end
    ζ̂_free = ζ̂
    ζ̂ = isempty(pin_idx) ? ζ̂ : expand_pinned(ζ̂, keep_idx, Nζ)
    β_q = β_ols[:, 1]
    β_Cp = β_ols[:, 2:end]
    β = β_q + β_Cp * ζ̂

    û = uq + uCp * ζ̂
    if return_vcov && n_pcs == 0 # with internal PCs, the vcov calculation is off.
        # Vcov routing rule for pairwise IV: all-pair complete-coverage CUE keeps
        # the maintained optimal-information formula. Incomplete CUE and any CUE
        # exclusion/pinning use the masked empirical sandwich with the full
        # candidate-dependent Jacobian. Fixed IV always uses the same frozen
        # entity/period bundle as its estimating moments.
        if isnothing(precisionvec)
            if complete_coverage && isempty(pin_idx) && !any(obs_index.exclpairs)
                σu²vec, Σζ = solve_optimal_vcov(ζ̂, û, S, C, obs_index)
            else
                σu²vec, Σζ = solve_vcov(û, S, C_free, uCp_free, obs_index;
                    ζ=ζ̂_free, complete_coverage=complete_coverage)
            end
        else
            # Fixed-weight SEs use the same frozen entity and period weights as the
            # estimating moments. Raw/custom one-step has `Mweights === nothing`;
            # feasible two-step carries the multipliers computed after step 1.
            σu²vec, Σζ = solve_vcov(û, S, C_free, uCp_free, obs_index; precision=precisionvec, Mweights=Mweights)
        end
        Σζ = isempty(pin_idx) ? Σζ : expand_pinned_vcov(Σζ, keep_idx, Nζ)
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

    dof = length(ζ̂) - length(pin_idx) + length(β)
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

@doc """
    giv(df, formula, id, t, weight; <keyword arguments>)

Estimate the granular instrumental-variables model

```math
q_{it} + p_t C_{it}'ζ = X_{it}'β + u_{it}
```

from panel data using cross-entity residual moment conditions and, under
complete coverage, market clearing.

# Arguments

- `df`: Input panel data.
- `formula`: A StatsModels formula of the form
  `q + interactions & endog(p) ~ exogenous_controls`. Use `fe(...)` for
  absorbed fixed effects and `pc(k)` for `k` common residual factors.
- `id`: Entity-identifier column.
- `t`: Time-identifier column. Together, `id` and `t` must identify rows.
- `weight`: Nonnegative entity-size or market-share column.

# Keyword arguments

- `precision_weights = :twostep`: Entity precision weighting. For `:iv` and
  `:iv_twopass`:
  - `:twostep` first solves with `1 / var(uq_i)` and equal period weights. At
    the first-step estimate `ζ̃`, it computes residual-based entity precisions
    and, under complete coverage, period multipliers. It freezes both weight
    families for one second solve. This is the default.
  - `:raw_onestep` performs only the first fixed-weight solve, with equal
    period weights.
  - `:cue` updates residual-based precisions and complete-coverage period
    multipliers at every candidate; this was the previous default.
  - An entity-length vector supplies custom fixed precisions in sorted entity
    order and uses equal period weights.
  Omitting this keyword selects `:twostep` and emits a one-time notice unless
  `quiet = true`. The quadratic and period-weight statements above apply to the
  IV algorithms; the specialized algorithms retain their own moment definitions.
- `guess = nothing`: Starting value for the endogenous coefficients. Accepts a
  number, a coefficient vector, or a dictionary keyed by coefficient name. OLS
  starting values are used when omitted.
- `algorithm = :iv`: Estimation algorithm. `:iv` is the standard estimator;
  `:iv_twopass` is its slower reference implementation. Both IV algorithms
  support either coverage regime. `:debiased_ols` and `:scalar_search` are
  separate estimators that require complete coverage.
- `exclude_pairs = Dict()`: Entity pairs to exclude from the moment conditions,
  supplied as `Dict(i => [j, ...])`.
- `pin_zero = String[]`: Exact endogenous coefficient names to fix at zero,
  dropping their moment rows so the reduced system remains exactly identified.
  Pinned entities remain in the panel and their covariance rows and columns are
  reported as zero. Scalar search does not support pinning; complete-coverage CUE
  with pins uses the general sandwich on the reduced system.
- `complete_coverage`: Required Boolean declaring whether the data design covers
  the full market. A `true` declaration is validated against market clearing,
  but adding-up alone is not used to infer coverage.
- `quiet = false`: Suppress informational messages and warnings.
- `save = :none`: Retain `:residuals`, `:fe`, `:all`, or neither (`:none`).
- `save_df = false`: Store the processed estimation data in the returned model.
- `return_vcov = true`: Compute variance estimates. Analytical standard errors
  are unavailable for specifications containing `pc(k)`.
- `contrasts = Dict()`: StatsModels contrast specifications.
- `tol = 1e-6`: Tolerance used by estimation and fixed-effect absorption.
- `iterations = 100`: Maximum solver iterations.
- `solver_options`: Additional options passed to NLsolve as a named tuple.
- `pca_option`: Options passed to HeteroPCA for specifications with `pc(k)`.

For fixed-weight IV, each solve is an exact quadratic system. Raw one-step,
custom weights, and feasible two-step use the standard sandwich covariance with
the weights held fixed in their estimating moments. The optimal covariance
formula is used only for CUE under complete coverage.

Under complete coverage, the economic period multiplier is
`M_t = 1 / ζS_t` on the maintained positive aggregate-elasticity domain. The
implementation uses `1 / clamp(abs(ζS_t), sqrt(eps), Inf)` only to keep
off-domain trial evaluations finite; a nonpositive or near-zero reported root
is marked non-converged.

# Returns

A `GIVModel`. The main result accessors are:

- `coef`, `stderror`, `vcov`, `confint`, and `coeftable` for the full model;
- `endog_coef`, `endog_vcov`, `exog_coef`, and `exog_vcov` for coefficient
  blocks;
- `agg_coef` for the aggregate or average elasticity.

Useful fields include `model.converged`, `model.coefdf`,
`model.residual_variance`, and, when requested, `model.df`, `model.fe`, and
`model.residual_df`.
""" giv

"""
    pin_zero_indices(pin_zero, endog_coefnames)

Validate exact endogenous coefficient names and return their sorted, unique
indices in `endog_coefnames`.
"""
function pin_zero_indices(pin_zero, endog_coefnames)
    isempty(pin_zero) && return Int[]
    idx = Int[]
    for name in pin_zero
        i = findfirst(==(String(name)), endog_coefnames)
        isnothing(i) && throw(ArgumentError(
            "`pin_zero` entry \"$name\" does not match any endogenous coefficient name; " *
            "available: $(join(endog_coefnames, ", "))"))
        push!(idx, i)
    end
    idx = sort!(unique(idx))
    length(idx) < length(endog_coefnames) ||
        throw(ArgumentError("`pin_zero` cannot pin every endogenous coefficient."))
    return idx
end

"Expand a free-coefficient vector to full length with exact zeros at pinned slots."
function expand_pinned(ζfree, keep_idx, Nζ)
    ζ = zeros(eltype(ζfree), Nζ)
    ζ[keep_idx] = ζfree
    return ζ
end

"Expand a free-coefficient covariance to full size with zero rows/columns at pinned slots."
function expand_pinned_vcov(Σfree, keep_idx, Nζ)
    Σ = zeros(eltype(Σfree), Nζ, Nζ)
    Σ[keep_idx, keep_idx] = Σfree
    return Σ
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
    check_market_clearing(q, S, obs_index)

Return `true` if the market-clearing (adding-up) condition is satisfied in every
nonempty period, and `false` otherwise.

This is a diagnostic and a validator for `complete_coverage=true`; passing the
check does not by itself establish that the sampled entities cover the whole
market.
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

For `build_error_function`, the default `precision_weights = :twostep` returns the
step-1 `:raw_onestep` moment map with precisions `1/var(uqᵢ)`, because this helper
does not run the second solve. Its period weights are therefore equal. Pass
`precision_weights = :cue` for the continuously updated map or an entity-length
vector for a custom fixed map. As with `giv`, `complete_coverage` is required.

"""
function build_error_function(df,
    formula::FormulaTerm,
    id::Symbol,
    t::Symbol,
    weight::Union{Symbol,Nothing}=nothing;
    exclude_pairs=Dict{Int,Vector{Int}}(),
    algorithm=:iv,
    quiet=false,
    complete_coverage::Bool,
    contrasts=Dict{Symbol,Any}(), # not tested;
    tol=1e-6,
    pca_option=(; impute_method=:zero, demean=false, maxiter=1000),
    precision_weights=:twostep,
    pin_zero=String[],
    kwargs...
)
    haskey(kwargs, :precision_mode) &&
        throw(ArgumentError("`precision_mode` was replaced by `precision_weights`; use :twostep, :raw_onestep, :cue, or an entity-length vector."))
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

    if complete_coverage && !check_market_clearing(q, S, obs_index)
        throw(ArgumentError("`complete_coverage=true` requires market clearing in every period, but the supplied data fail the adding-up check."))
    end
    if algorithm ∈ [:scalar_search, :debiased_ols] && !complete_coverage
        throw(ArgumentError("`algorithm=$(algorithm)` requires `complete_coverage=true`."))
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

        isempty(pin_zero) ||
            throw(ArgumentError("`pin_zero` is not supported with `algorithm = :scalar_search`."))
        err_func = x -> ζS_err(x, uqmat, p, S_vec, coefmapping; kwargs...)
        return err_func, (uqmat=uqmat, p=p, S_vec=S_vec, coefmapping=coefmapping)
    else
        # :twostep resolves to the step-1 raw weights here (no solve loop in this export).
        precisionvec = resolve_precision(precision_weights, uq, obs_index)
        pin_idx = pin_zero_indices(pin_zero, endog_coefnames)
        keep_idx = setdiff(1:size(C, 2), pin_idx)
        C = isempty(pin_idx) ? C : C[:, keep_idx]
        uCp = isempty(pin_idx) ? uCp : uCp[:, keep_idx]
        err_func = x -> mean_moment_conditions(x, uq, uCp, C, S, obs_index, complete_coverage, Val{algorithm}(), n_pcs, pca_option; precision=precisionvec)
        # Exact Jacobian of the exported moment map, when defined (fixed precisions,
        # `:iv`/`:iv_twopass`, no internal PCs); `nothing` otherwise.
        jac_func = (!isnothing(precisionvec) && n_pcs == 0 && algorithm in (:iv, :iv_twopass)) ?
                   (x -> mean_moment_jacobian(x, uq, uCp, C, S, obs_index, complete_coverage, precisionvec)) :
                   nothing
        return err_func, (uq=uq, uCp=uCp, C=C, S=S, obs_index=obs_index, n_pcs=n_pcs, precision=precisionvec, jac_func=jac_func)
    end
end
