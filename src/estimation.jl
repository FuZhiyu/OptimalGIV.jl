function estimate_giv(
    q,
    Cp,
    C,
    S,
    obs_index,
    ::A;
    guess=nothing,
    quiet=false,
    complete_coverage::Bool,
    solver_options=(;),
    n_pcs=0,
    pca_option=(; impute_method=:zero, demean=false, maxiter=1000),
    precision=nothing,
    Mweights=nothing,
) where {A<:Union{Val{:iv},Val{:iv_twopass},Val{:debiased_ols}}}
    if isnothing(guess)
        if !quiet
            @info "Initial guess is not provided. Using OLS estimate as initial guess."
        end
        guess = (Cp' * Cp) \ (Cp' * q)
    end

    Nmom = size(Cp, 2)
    fmom = x -> mean_moment_conditions(x, q, Cp, C, S, obs_index, complete_coverage, A(), n_pcs, pca_option; precision=precision, Mweights=Mweights)
    err0 = fmom(guess)
    if length(err0) != Nmom
        throw(ArgumentError("The number of moment conditions is not equal to the number of initial guess."))
    end
    # Guard the INITIAL evaluation only: this is exactly where NLsolve would raise
    # its opaque `IsFiniteException` (`check_isfinite` on the initial residual). We
    # replace it with a named diagnosis and leave NLsolve's mid-solve recovery
    # (step rejection on Inf trials, graceful non-converged return on NaN trials)
    # untouched — no default-behavior change beyond the sanctioned error swap.
    if any(!isfinite, err0)
        throw(diagnose_nonfinite_at_guess(guess, q, Cp, C, S, obs_index; precision=precision))
    end

    # Analytic Jacobian: exact for the fixed-precision `:iv`/`:iv_twopass` kernels
    # (`:raw_onestep`, custom vectors, and both `:twostep` steps), where residuals
    # are linear in ζ and momweight is constant. CUE (`precision === nothing`), internal PCs
    # (loadings move with ζ), and `:debiased_ols` keep the autodiff/FD path.
    use_analytic = !isnothing(precision) && n_pcs == 0 &&
                   A <: Union{Val{:iv},Val{:iv_twopass}}
    res = if use_analytic
        # NLsolve rejects `autodiff` when an explicit Jacobian is supplied. Keep
        # all other NLsolve controls, including `method`, on the single
        # `solver_options` path.
        analytic_options = (; (k => v for (k, v) in pairs(solver_options) if k != :autodiff)...)
        nlsolve(
            fmom,
            x -> mean_moment_jacobian(x, q, Cp, C, S, obs_index, complete_coverage, precision; Mweights=Mweights),
            guess;
            analytic_options...,
        )
    else
        nlsolve(fmom, guess; solver_options...)
    end

    ζ̂ = res.zero
    converged = res.f_converged
    if complete_coverage && !aggregate_elasticity_in_domain(ζ̂, C, S, obs_index)
        converged = false
        !quiet && @warn "The reported root has nonpositive or near-zero aggregate elasticity under complete coverage."
    end

    if !converged && !quiet
        @warn "The estimation did not converge."
    end

    return ζ̂, converged
end

function moment_conditions(ζ, q, Cp, C, S, obs_index, complete_coverage, ::Val{:iv_twopass}, n_pcs=0, pca_option=(; impute_method=:zero, demean=false, maxiter=1000); precision=nothing, Mweights=nothing)
    Nmom = length(ζ)
    N, T = obs_index.N, obs_index.T


    # Calculate residuals
    u = q + Cp * ζ

    # Extract PCs and update residuals if requested
    loadings_matrix = Matrix{eltype(ζ)}(undef, N, n_pcs)  # N×n_pcs matrix for type stability
    if n_pcs > 0
        _, loadings_matrix, _, _ = extract_pcs_from_residuals(u, obs_index, n_pcs; pca_option...)
    end

    # Entity precision weights. CUE (`precision === nothing`): update with ζ via
    # residual variance. Fixed-weight mode: use the supplied data-based precisions.
    prec = isnothing(precision) ? 1 ./ calculate_entity_variance(u, obs_index) : precision

    # Initialize error and weightsum
    err = zeros(eltype(ζ), Nmom, T)
    weightsum = zeros(eltype(ζ), Nmom, T)

    # Loop through time periods
    @threads for t in 1:T
        # Get observations range for this time period
        start_idx = obs_index.start_indices[t]
        end_idx = obs_index.end_indices[t]

        for imom in 1:Nmom
            # First loop: iterate over i, then j > i
            # Do it in two loops so that skipping based on iszero(C) check is more efficient
            for idx_i in start_idx:end_idx
                i = obs_index.ids[idx_i]

                # Skip if C is zero
                if iszero(C[idx_i, imom])
                    continue
                end

                weight_i = C[idx_i, imom] * prec[i]

                for idx_j in (idx_i+1):end_idx
                    j = obs_index.ids[idx_j]

                    # Since idx_i < idx_j within a time period, we know i ≠ j
                    if obs_index.exclpairs[i, j]
                        continue
                    end

                    empirical_cov = u[idx_i] * u[idx_j]
                    expected_cov = n_pcs > 0 ? dot(loadings_matrix[i, :], loadings_matrix[j, :]) : zero(eltype(ζ))
                    err[imom, t] += (empirical_cov - expected_cov) * weight_i * S[idx_j]
                    weightsum[imom, t] += weight_i * S[idx_j]
                end

            end

            # Second loop: iterate over j, then i < j
            for idx_j in start_idx:end_idx
                j = obs_index.ids[idx_j]

                # Skip if C is zero
                if iszero(C[idx_j, imom])
                    continue
                end

                weight_j = C[idx_j, imom] * prec[j]

                for idx_i in start_idx:(idx_j-1)
                    i = obs_index.ids[idx_i]

                    # Since idx_i < idx_j within a time period, we know i ≠ j
                    if obs_index.exclpairs[i, j]
                        continue
                    end

                    empirical_cov = u[idx_i] * u[idx_j]
                    expected_cov = n_pcs > 0 ? dot(loadings_matrix[i, :], loadings_matrix[j, :]) : zero(eltype(ζ))
                    err[imom, t] += (empirical_cov - expected_cov) * weight_j * S[idx_i]
                    weightsum[imom, t] += weight_j * S[idx_i]
                end

            end
        end
    end

    # the efficient weighting requires the scaling of multiplier for each period
    # only feasible when we observe the full market
    period_weights = resolve_period_mweights(ζ, C, S, obs_index, complete_coverage, precision, Mweights)
    if !isnothing(period_weights)
        err .*= period_weights'
        # weightsum is left unscaled
    end

    # equal weight the final moment conditions for numerical stability
    # it's exactly identified at this point hence it's not affecting the estimation
    momweight = sum(abs.(weightsum); dims=2)
    momweight ./= sum(momweight)
    err ./= momweight

    return err
end


function moment_conditions(ζ, q, Cp, C, S, obs_index, complete_coverage, ::Val{:iv}, n_pcs=0, pca_option=(; impute_method=:zero, demean=false, maxiter=1000); precision=nothing, Mweights=nothing)

    Nm = length(ζ)
    N, T = obs_index.N, obs_index.T
    err = zeros(eltype(ζ), Nm, T)

    # residuals and entity-level precision ------------------------------
    u = q .+ Cp * ζ

    # Extract PCs and update residuals if requested
    loadings_matrix = Matrix{eltype(ζ)}(undef, N, n_pcs)  # N×n_pcs matrix for type stability
    if n_pcs > 0
        _, loadings_matrix, _, _ = extract_pcs_from_residuals(u, obs_index, n_pcs; pca_option...)
    end

    # Entity precision weights. CUE (`precision === nothing`): update with ζ via
    # residual variance. Fixed-weight mode: use the supplied data-based precisions.
    prec = isnothing(precision) ? inv.(calculate_entity_variance(u, obs_index)) : precision

    weightsum = zeros(eltype(ζ), Nm, T)
    # 1️⃣ fast O(N) pass
    fast_pass!(weightsum, err, u, C, S, prec, obs_index, loadings_matrix, n_pcs)

    # 2️⃣ subtract excluded pairs
    deduct_excluded_pairs!(err, weightsum, C, S, u, prec, obs_index, loadings_matrix, n_pcs)

    # the efficient weighting requires the scaling of multiplier for each period
    # only feasible when we observe the full market
    period_weights = resolve_period_mweights(ζ, C, S, obs_index, complete_coverage, precision, Mweights)
    if !isnothing(period_weights)
        err .*= period_weights'
        # weightsum is left unscaled
    end

    # equal weight the final moment conditions for numerical stability
    # it's exactly identified at this point hence it's not affecting the estimation
    momweight = sum(abs.(weightsum); dims=2)
    momweight ./= sum(momweight)
    err ./= momweight

    return err
end

# ----------------------------------------------------------------------
#  FAST  O(N)  PASS   (fills only `err`)
# ----------------------------------------------------------------------
function fast_pass!(weightsum, err, u, C, S, prec, obs_index, loadings_matrix=Matrix{eltype(err)}(undef, 0, 0), n_pcs=0)
    Nm = size(err, 1)
    T = obs_index.T

    # Parallelize over time periods
    @threads for t in 1:T
        r = obs_index.start_indices[t]:obs_index.end_indices[t]
        ids_t = obs_index.ids[r]

        u_t = @view u[r]
        S_t = @view S[r]
        prec_t = prec[ids_t]

        # Pre-compute these values once per time period
        b_total = sum(S_t .* u_t)
        S_total = sum(S_t)

        # Local buffer for non-zero indices to avoid reallocations
        nz_buffer = BitVector(undef, length(r))

        for m in 1:Nm
            C_t = @view C[r, m]

            # Reuse buffer for non-zero indices
            @inbounds for i in eachindex(C_t)
                nz_buffer[i] = C_t[i] != 0
            end

            # Skip if all zeros
            if !any(nz_buffer)
                continue
            end

            # Use @inbounds for inner loop operations
            @inbounds begin
                # Compute these vectors once per moment condition
                weight_vec = C_t[nz_buffer] .* prec_t[nz_buffer]
                Su_nz = S_t[nz_buffer] .* u_t[nz_buffer]

                # First compute error values using fast O(N) formula
                a_vec = weight_vec .* u_t[nz_buffer]
                diag_ab = a_vec .* Su_nz

                # Fused operations for better performance
                err[m, t] = sum(a_vec) * b_total - sum(diag_ab)

                # Factor correction if n_pcs > 0
                if n_pcs > 0
                    # Get loadings for entities in this time period
                    loadings_t = loadings_matrix[ids_t, :]  # length(r) × n_pcs

                    # Compute factor correction terms for each PC
                    for k in 1:n_pcs
                        loadings_k = loadings_t[:, k]
                        loadings_k_nz = loadings_k[nz_buffer]

                        # Total loadings for this factor across all entities
                        total_loadings_k = sum(S_t .* loadings_k)

                        # Factor correction using mathematical identity
                        factor_correction_k = sum(weight_vec .* loadings_k_nz) * total_loadings_k -
                                              sum(weight_vec .* S_t[nz_buffer] .* loadings_k_nz .^ 2)

                        err[m, t] -= factor_correction_k
                    end
                end

                # Calculate weightsum
                sum_weight = sum(weight_vec)
                diag_ws = weight_vec .* S_t[nz_buffer]

                # Compute weightsum in O(N) time
                weightsum[m, t] = sum_weight * S_total - sum(diag_ws)
            end
        end
    end

    return weightsum
end


# ----------------------------------------------------------------------
#  SECOND PASS  – deduct excluded pairs from err and weightsum
# ----------------------------------------------------------------------
function deduct_excluded_pairs!(err, weightsum, C, S, u, prec, obs_index, loadings_matrix=Matrix{eltype(err)}(undef, 0, 0), n_pcs=0)
    Nmom = size(err, 1)
    T = obs_index.T

    # Loop through time periods
    @threads for t in 1:T
        # Get observations range for this time period
        start_idx = obs_index.start_indices[t]
        end_idx = obs_index.end_indices[t]

        for imom in 1:Nmom
            # First loop: iterate over i, then j > i
            for idx_i in start_idx:end_idx
                i = obs_index.ids[idx_i]

                # Skip if C is zero
                if iszero(C[idx_i, imom])
                    continue
                end

                weight_i = C[idx_i, imom] * prec[i]

                for idx_j in (idx_i+1):end_idx
                    j = obs_index.ids[idx_j]

                    # We only want to deduct excluded pairs
                    if !obs_index.exclpairs[i, j]
                        continue
                    end

                    # Compute empirical covariance
                    empirical_cov = u[idx_i] * u[idx_j]

                    # Compute expected covariance from factors if n_pcs > 0
                    expected_cov = zero(eltype(empirical_cov))
                    if n_pcs > 0
                        expected_cov = dot(loadings_matrix[i, :], loadings_matrix[j, :])
                    end

                    # Deduct contribution for excluded pairs (accounting for factor structure)
                    err[imom, t] -= (empirical_cov - expected_cov) * weight_i * S[idx_j]
                    weightsum[imom, t] -= weight_i * S[idx_j]
                end
            end

            # Second loop: iterate over j, then i < j
            for idx_j in start_idx:end_idx
                j = obs_index.ids[idx_j]

                # Skip if C is zero
                if iszero(C[idx_j, imom])
                    continue
                end

                weight_j = C[idx_j, imom] * prec[j]

                for idx_i in start_idx:(idx_j-1)
                    i = obs_index.ids[idx_i]

                    # We only want to deduct excluded pairs
                    if !obs_index.exclpairs[i, j]
                        continue
                    end

                    # Compute empirical covariance
                    empirical_cov = u[idx_i] * u[idx_j]

                    # Compute expected covariance from factors if n_pcs > 0
                    expected_cov = zero(eltype(empirical_cov))
                    if n_pcs > 0
                        expected_cov = dot(loadings_matrix[i, :], loadings_matrix[j, :])
                    end

                    # Deduct contribution for excluded pairs (accounting for factor structure)
                    err[imom, t] -= (empirical_cov - expected_cov) * weight_j * S[idx_i]
                    weightsum[imom, t] -= weight_j * S[idx_i]
                end
            end
        end
    end

    return nothing
end


function moment_conditions(ζ, q, Cp, C, S, obs_index, complete_coverage, ::Val{:debiased_ols}, n_pcs=0, pca_option=(; impute_method=:zero, demean=false, maxiter=1000); precision=nothing, Mweights=nothing)
    if n_pcs > 0
        throw(ArgumentError("PC extraction (n_pcs > 0) is not yet supported for the :debiased_ols algorithm. Use :iv_twopass instead."))
    end

    Nmom = length(ζ)
    T = obs_index.T

    # Calculate residuals
    u = q + Cp * ζ

    # Entity precision weights. CUE (`precision === nothing`): update with ζ via
    # residual variance (normalized). Fixed-weight mode: use the supplied precisions.
    prec = isnothing(precision) ? (1 ./ calculate_entity_variance(u, obs_index)) : copy(precision)
    prec ./= sum(prec)

    # Initialize error and weightsum
    err = zeros(eltype(ζ), Nmom, T)
    weightsum = zeros(eltype(ζ), Nmom, T)

    # Calculate ζS for each time period
    ζSvec = solve_aggregate_elasticity(ζ, C, S, obs_index; complete_coverage=true)

    # Loop through time periods
    @threads for t in 1:T
        # Get observations range for this time period
        start_idx = obs_index.start_indices[t]
        end_idx = obs_index.end_indices[t]

        for imom in 1:Nmom
            for idx in start_idx:end_idx
                i = obs_index.ids[idx]

                # Skip if C is zero
                if iszero(C[idx, imom])
                    continue
                end

                weight = prec[i]
                uCp = u[idx] * Cp[idx, imom]
                CSσ² = u[idx]^2 * C[idx, imom] * S[idx]
                # alternatively, use estimated variance
                # CSσ² = σu²vec[i] * C[idx, imom] * S[idx]

                err[imom, t] += weight * (uCp - CSσ² / ζSvec[t])
                weightsum[imom, t] += weight
            end
        end
    end

    # Equal weight the moment conditions for numerical stability
    momweight = sum(abs.(weightsum); dims=2)
    momweight ./= sum(momweight)
    err ./= momweight

    return err
end


mean_moment_conditions(ζ, q, Cp, C, S, obs_index, complete_coverage, algorithm, n_pcs=0, pca_option=(; impute_method=:zero, demean=false, maxiter=1000); precision=nothing, Mweights=nothing) =
    vec(mean(moment_conditions(ζ, q, Cp, C, S, obs_index, complete_coverage, algorithm, n_pcs, pca_option; precision=precision, Mweights=Mweights); dims=2))

# ----------------------------------------------------------------------
#  ANALYTIC JACOBIAN — fixed-precision moment kernel
# ----------------------------------------------------------------------
"""
    mean_moment_jacobian(ζ, q, Cp, C, S, obs_index, complete_coverage, precision; Mweights=nothing)

Exact Jacobian `J[m, k] = ∂gₘ/∂ζₖ` of the fixed-precision mean moment map
(`mean_moment_conditions` under `:iv`/`:iv_twopass` with `precision !== nothing`).

With the entity precisions held fixed, the residuals `u = q + Cp ζ` are linear
in ζ, so `∂(uᵢuⱼ)/∂ζₖ = Cpᵢₖ uⱼ + uᵢ Cpⱼₖ` and the Jacobian has the same
pair-sum structure as the moments: the O(N) fast-pass identity applies verbatim
(products of per-period aggregates), followed by the excluded-pair deduction.
The per-moment normalization `momweight` depends only on `(precision, C, S)`,
so it is a constant row scaling. Optional frozen `Mweights` scale each period's
Jacobian contribution but are constant with respect to the current `ζ`, so no
period-weight product-rule term enters the fixed-weight Jacobian.

Not valid for CUE (`precision === nothing`) or internal PC extraction inside
the kernel (`n_pcs > 0` with ζ-dependent loadings) — there the weights/loadings
move with ζ and the solver keeps the autodiff/FD path.
"""
function mean_moment_jacobian(ζ, q, Cp, C, S, obs_index, complete_coverage, precision; Mweights=nothing)
    isnothing(precision) &&
        throw(ArgumentError("mean_moment_jacobian requires fixed precisions; CUE (`precision === nothing`) has no closed-form Jacobian here."))
    prec = precision
    Nm = length(ζ)
    T = obs_index.T
    u = q .+ Cp * ζ
    TT = eltype(u)
    loadings = Matrix{TT}(undef, obs_index.N, 0)

    # Raw moment levels and weightsum: weightsum fixes the constant row scaling.
    err = zeros(TT, Nm, T)
    weightsum = zeros(TT, Nm, T)
    fast_pass!(weightsum, err, u, C, S, prec, obs_index, loadings, 0)
    deduct_excluded_pairs!(err, weightsum, C, S, u, prec, obs_index, loadings, 0)

    has_excl = any(obs_index.exclpairs)

    # ∂err[m,t]/∂ζₖ by the same O(N) identity as the moments (vᵏ = Cp[:, k], wᵢ = Cᵢₘ precᵢ):
    #   Σ_{i≠j} wᵢ (vᵢᵏ uⱼ + uᵢ vⱼᵏ) Sⱼ
    #     = (Σᵢ wᵢvᵢᵏ)(Σⱼ Sⱼuⱼ) + (Σᵢ wᵢuᵢ)(Σⱼ Sⱼvⱼᵏ) − 2 Σᵢ wᵢSᵢuᵢvᵢᵏ
    Jt = zeros(TT, Nm, Nm, T)
    @threads for t in 1:T
        r = obs_index.start_indices[t]:obs_index.end_indices[t]
        ids_t = obs_index.ids[r]
        u_t = view(u, r)
        S_t = view(S, r)
        Cp_t = view(Cp, r, :)
        prec_t = prec[ids_t]
        bu = dot(S_t, u_t)
        bv = Cp_t' * S_t

        for m in 1:Nm
            C_tm = view(C, r, m)
            nz = findall(!iszero, C_tm)
            isempty(nz) && continue
            w = C_tm[nz] .* prec_t[nz]
            u_nz = u_t[nz]
            wu_sum = dot(w, u_nz)
            wSu = w .* S_t[nz] .* u_nz
            for k in 1:Nm
                v_nz = view(Cp_t, nz, k)
                Jt[m, k, t] = dot(w, v_nz) * bu + wu_sum * bv[k] - 2 * dot(wSu, v_nz)
            end
        end

        # deduct excluded pairs (both orderings of each pair, as in the moments)
        if has_excl
            for idx_i in r
                i = obs_index.ids[idx_i]
                for idx_j in (idx_i+1):last(r)
                    j = obs_index.ids[idx_j]
                    obs_index.exclpairs[i, j] || continue
                    for m in 1:Nm
                        cw = C[idx_i, m] * prec[i] * S[idx_j] + C[idx_j, m] * prec[j] * S[idx_i]
                        iszero(cw) && continue
                        for k in 1:Nm
                            Jt[m, k, t] -= cw * (Cp[idx_i, k] * u[idx_j] + u[idx_i] * Cp[idx_j, k])
                        end
                    end
                end
            end
        end
    end

    momweight = vec(sum(abs.(weightsum); dims=2))
    momweight ./= sum(momweight)

    J = zeros(TT, Nm, Nm)
    if !isnothing(Mweights)
        for k in 1:Nm, m in 1:Nm
            acc = zero(TT)
            for t in 1:T
                acc += Mweights[t] * Jt[m, k, t]
            end
            J[m, k] = acc / (T * momweight[m])
        end
    else
        for k in 1:Nm, m in 1:Nm
            J[m, k] = sum(view(Jt, m, k, :)) / (T * momweight[m])
        end
    end
    return J
end

function solve_aggregate_elasticity(ζ, C, S, obs_index; complete_coverage::Bool)
    Nmom = length(ζ)
    ζSvec = zeros(eltype(ζ), obs_index.T)

    @views for t in 1:obs_index.T
        r = obs_index.start_indices[t]:obs_index.end_indices[t]
        ζSvec[t] = dot(S[r], C[r, :] * ζ)
        if !complete_coverage # when we do not have the whole market, report avg instead
            ζSvec[t] /= sum(S[r])
        end
    end

    return ζSvec
end

"""
    period_mweights(ζ, C, S, obs_index)

Clamped, normalized period weights applied to the moments under complete
coverage: `Mweights_t ∝ 1 / clamp(|ζS_t|, √eps, Inf)`, normalized to sum to one.
Shared between the `:iv`/`:iv_twopass` moment kernels and the sandwich `solve_vcov`
so the vcov `W` carries exactly the period scaling of the moments actually solved
(period weighting is column scaling *before* averaging over `t` — it changes the
estimator, unlike the `momweight` row scaling, which cancels in the sandwich).
"""
function period_mweights(ζ, C, S, obs_index)
    ζS = solve_aggregate_elasticity(ζ, C, S, obs_index; complete_coverage=true)
    Mweights = 1 ./ clamp.(abs.(ζS), sqrt(eps(eltype(ζS))), Inf) # avoid division by zero
    Mweights ./= sum(Mweights)
    return Mweights
end

resolve_period_mweights(ζ, C, S, obs_index, complete_coverage, precision, Mweights) =
    !complete_coverage ? nothing : isnothing(precision) ? period_mweights(ζ, C, S, obs_index) : Mweights

function aggregate_elasticity_in_domain(ζ, C, S, obs_index)
    ζS = solve_aggregate_elasticity(ζ, C, S, obs_index; complete_coverage=true)
    return all(>(sqrt(eps(eltype(ζS)))), ζS)
end

"""
    admissible_pair_indices(obs_index)

Return the co-occurring unordered entity pairs retained by the estimator's
exclusion mask. Both sandwich scores and the complete-coverage optimal route use
this pair catalog so covariance construction cannot silently restore excluded
moments.
"""
function admissible_pair_indices(obs_index)
    present = obs_index.entity_obs_indices .> 0
    cooccurs = present * transpose(present)
    inds = findall(triu((cooccurs .> zero(eltype(cooccurs))) .& .!obs_index.exclpairs, 1))
    return [I[1] for I in inds], [I[2] for I in inds]
end

function solve_optimal_vcov(ζ, u, S, C, obs_index)
    any(obs_index.exclpairs) && throw(ArgumentError(
        "`solve_optimal_vcov` requires the all-pair complete-coverage CUE design; " *
        "excluded pairs must use the masked empirical sandwich."))
    Nmom = length(ζ)
    N, T = obs_index.N, obs_index.T
    σu²vec = calculate_entity_variance(u, obs_index)
    ζSvec = solve_aggregate_elasticity(ζ, C, S, obs_index; complete_coverage=true)
    Mvec = 1 ./ ζSvec

    # Step 1: identify the estimator's admissible co-occurring pairs. The guard
    # above makes this the maintained all-pair route, while sharing the catalog
    # implementation with the masked sandwich.
    pair_i, pair_j = admissible_pair_indices(obs_index)
    # Number of unique entity pairs
    n_pairs = length(pair_i)

    # Step 2: Initialize arrays for computation
    Vdiag = zeros(n_pairs)
    D = zeros(n_pairs, Nmom, T)

    # Step 3: Compute Vdiag for all pairs
    for idx in 1:n_pairs
        i, j = pair_i[idx], pair_j[idx]
        Vdiag[idx] = σu²vec[i] * σu²vec[j]
    end

    # Step 4: Compute D matrix efficiently using pre-stored observation indices
    for t in 1:T
        # Process all relevant pairs for this time period
        for idx in 1:n_pairs
            i, j = pair_i[idx], pair_j[idx]

            # Get observation indices directly from the matrix
            i_pos = obs_index.entity_obs_indices[i, t]
            j_pos = obs_index.entity_obs_indices[j, t]

            # Skip if either entity is not in this period
            if i_pos == 0 || j_pos == 0
                continue
            end

            for imom in 1:Nmom
                D[idx, imom, t] =
                    σu²vec[j] * S[j_pos] * C[i_pos, imom] +
                    σu²vec[i] * S[i_pos] * C[j_pos, imom]
                D[idx, imom, t] *= Mvec[t]
            end
        end
    end

    # Step 5: Final calculation
    Vinv = inv(Diagonal(Vdiag))
    DVinvD = mean([D[:, :, t]' * Vinv * D[:, :, t] for t in 1:T])
    Σζ = inv(DVinvD) / T

    return σu²vec, Σζ
end

"""
    masked_period_scores(u, S, C, Cp, obs_index, precision, Mweights; bread=false)

Construct the actual period scores from the estimator's admissible pairs and
weight bundle. `precision` and `Mweights` are already resolved for the candidate:
fixed IV passes its frozen bundle, while CUE passes candidate-updated values. The
returned score matrix includes the same period and `momweight` normalization as
the IV moment kernel. With `bread=true`, also return the exact fixed-weight
Jacobian in moment-by-parameter orientation, normalized by `1 / T`.
"""
function masked_period_scores(u, S, C, Cp, obs_index, precision, Mweights; bread=false)
    Nmom = size(C, 2)
    T = obs_index.T
    pair_i, pair_j = admissible_pair_indices(obs_index)
    n_pairs = length(pair_i)
    TT = promote_type(eltype(u), eltype(S), eltype(C), eltype(Cp), eltype(precision))
    scores = zeros(TT, Nmom, T)
    weightsum = zeros(TT, Nmom, T)
    Gt = bread ? zeros(TT, Nmom, Nmom, T) : nothing

    for t in 1:T
        Wt = zeros(TT, n_pairs, Nmom)
        ht = zeros(TT, n_pairs)
        Dt = bread ? zeros(TT, n_pairs, Nmom) : nothing
        for idx in 1:n_pairs
            i, j = pair_i[idx], pair_j[idx]
            i_pos = obs_index.entity_obs_indices[i, t]
            j_pos = obs_index.entity_obs_indices[j, t]
            (i_pos == 0 || j_pos == 0) && continue
            ht[idx] = u[i_pos] * u[j_pos]
            for k in 1:Nmom
                Wt[idx, k] = precision[i] * S[j_pos] * C[i_pos, k] +
                             precision[j] * S[i_pos] * C[j_pos, k]
                if bread
                    Dt[idx, k] = Cp[i_pos, k] * u[j_pos] + u[i_pos] * Cp[j_pos, k]
                end
            end
        end
        scores[:, t] .= Wt' * ht
        weightsum[:, t] .= vec(sum(Wt; dims=1))
        bread && (Gt[:, :, t] .= Wt' * Dt)
    end

    momweight = vec(sum(abs.(weightsum); dims=2))
    momweight ./= sum(momweight)
    for t in 1:T
        period_weight = isnothing(Mweights) ? one(TT) : Mweights[t]
        scores[:, t] .*= period_weight ./ momweight
        if bread
            @views Gt[:, :, t] .*= period_weight
            @views Gt[:, :, t] ./= momweight
        end
    end

    G = bread ? dropdims(mean(Gt; dims=3); dims=3) : nothing
    return scores, G
end

function _central_difference_jacobian(f, x)
    fx = f(x)
    J = zeros(promote_type(eltype(fx), eltype(x)), length(fx), length(x))
    for k in eachindex(x)
        h = cbrt(eps(float(eltype(x)))) * max(abs(x[k]), one(eltype(x)))
        xp, xm = copy(x), copy(x)
        xp[k] += h
        xm[k] -= h
        J[:, k] .= (f(xp) - f(xm)) ./ (2h)
    end
    return J
end

"""
    solve_vcov(u, S, C, Cp, obs_index; ζ=nothing, complete_coverage=nothing,
               precision=nothing, Mweights=nothing)

Masked empirical sandwich for the pairwise IV estimating equation. Fixed-weight
routes use the exact affine Jacobian and the supplied frozen entity/period weight
bundle. CUE routes require `ζ` and `complete_coverage`; their bread is a centered
finite-difference derivative of the full candidate-dependent score map, including
entity precisions, normalized complete-coverage period weights, and `momweight`.
The meat is the empirical outer-product average of the centered masked period
scores (centering is immaterial at an exact root and protects approximate solves).
"""
function solve_vcov(u, S, C, Cp, obs_index;
    ζ=nothing,
    complete_coverage=nothing,
    precision=nothing,
    Mweights=nothing,
)
    T = obs_index.T
    σu²vec = calculate_entity_variance(u, obs_index)

    G, scores = if isnothing(precision)
        (isnothing(ζ) || isnothing(complete_coverage)) && throw(ArgumentError(
            "CUE sandwich inference requires both `ζ` and `complete_coverage` so the full candidate-dependent Jacobian can be evaluated."))
        candidate_scores = function (z)
            uz = u + Cp * (z - ζ)
            candidate_precision = 1 ./ calculate_entity_variance(uz, obs_index)
            candidate_Mweights = complete_coverage ? period_mweights(z, C, S, obs_index) : nothing
            first(masked_period_scores(uz, S, C, Cp, obs_index,
                candidate_precision, candidate_Mweights))
        end
        score_matrix = candidate_scores(ζ)
        moment_map = z -> vec(mean(candidate_scores(z); dims=2))
        _central_difference_jacobian(moment_map, ζ), score_matrix
    else
        score_matrix, fixed_G = masked_period_scores(u, S, C, Cp, obs_index,
            precision, Mweights; bread=true)
        fixed_G, score_matrix
    end

    centered_scores = scores .- mean(scores; dims=2)
    B = centered_scores * centered_scores' / T
    invG = inv(G)
    Σζ = invG * B * invG' / T
    return σu²vec, Symmetric(Σζ + Σζ') / 2
end

"""
Solve the OLS covariance matrix for the GIV estimation.

This version uses the observation indexing structure instead of matrices directly.

Parameters:
- σu²vec: Pre-computed entity-specific residual variances
- X: Matrix of exogenous variables
- obs_index: Observation index structure with entity mappings

Returns:
- Covariance matrix for OLS estimator
"""
function solve_ols_vcov(σu²vec, X, obs_index)
    N, T = obs_index.N, obs_index.T
    Nmom = size(X, 2)

    # Preallocate the final matrices
    bread = zeros(Nmom, Nmom)
    meat = zeros(Nmom, Nmom)

    # Process a chunk of entities
    function process_chunk(chunk_range)::Tuple{Matrix{Float64},Matrix{Float64}}
        local_bread = zeros(Nmom, Nmom)
        local_meat = zeros(Nmom, Nmom)

        for i in chunk_range
            # Collect X values for this entity
            X_i = zeros(0, Nmom)
            n_obs = 0  # Count observations for normalization

            # Find all observations for entity i
            for t in 1:T
                # Get observation index for entity i in time period t
                idx = obs_index.entity_obs_indices[i, t]

                # Skip if entity is not in this period
                if idx == 0
                    continue
                end

                # Add to collected values
                X_i = vcat(X_i, transpose(X[idx, :]))
                n_obs += 1
            end

            # Skip if not enough observations
            if n_obs < 2
                continue
            end

            # Identify non-zero columns to exploit sparsity
            zero_cols = vec(all(iszero, X_i; dims=1))
            nonzero_cols = findall(!x -> x, zero_cols)
            if isempty(nonzero_cols)
                continue  # Skip if all columns are zero
            end

            # Extract non-zero columns
            X_i_nonzero = X_i[:, nonzero_cols]

            # Compute XX'_i matrix
            XX_i_sub = zeros(length(nonzero_cols), length(nonzero_cols))
            BLAS.syrk!('U', 'T', 1.0 / n_obs, X_i_nonzero, 0.0, XX_i_sub)

            # Wrap with Symmetric to represent the full symmetric matrix
            XX_i_sub_sym = Symmetric(XX_i_sub, :U)

            # Map back to original dimensions
            full_XX_i = zeros(Nmom, Nmom)
            full_XX_i[nonzero_cols, nonzero_cols] .= XX_i_sub_sym

            # Update local accumulators
            local_bread .+= full_XX_i
            local_meat .+= full_XX_i .* σu²vec[i]
        end

        return local_bread, local_meat
    end

    # Create smaller chunks for better load balancing
    nthreads = Threads.nthreads()
    # Use more chunks than threads for better load balancing
    n_chunks = nthreads * 2
    chunk_size = cld(N, n_chunks)

    # Create and spawn tasks with type annotation
    tasks = Vector{Task}()
    for c in 1:n_chunks
        start_idx = (c - 1) * chunk_size + 1
        end_idx = min(c * chunk_size, N)
        if start_idx <= end_idx
            task = Threads.@spawn process_chunk(start_idx:end_idx)
            push!(tasks, task)
        end
    end

    # Collect results from all tasks
    for task in tasks
        thread_bread, thread_meat = fetch(task)::Tuple{Matrix{Float64},Matrix{Float64}}
        bread .+= thread_bread
        meat .+= thread_meat
    end

    # Compute the covariance matrix using Symmetric to exploit symmetry
    bread_sym = Symmetric(bread)
    bread_inv = inv(bread_sym)
    vcov_ols = bread_inv * meat * bread_inv / T

    return vcov_ols
end

"""
Calculate variance by entity for zero-mean residuals.

Since we know the residuals have zero mean by construction, this function
simply computes the sum of squared values divided by the observation count.

Parameters:
- u: Vector of residuals (with zero mean)
- obs_index: Observation index structure with entity mappings

Returns:
- Vector of variances for each entity

Throws:
- ArgumentError if any entity has fewer than 2 observations
"""
function calculate_entity_variance(u, obs_index)
    N = obs_index.N
    σu²vec = zeros(eltype(u), N)
    counts = zeros(Int, N)

    # Sum of squares for zero-mean data
    for idx in 1:length(u)
        entity = obs_index.ids[idx]
        σu²vec[entity] += u[idx]^2
        counts[entity] += 1
    end

    # Divide by count - 1 for each entity
    for i in 1:N
        σu²vec[i] /= (counts[i] - 1)
    end

    return σu²vec
end

# ----------------------------------------------------------------------
#  NONFINITE-MOMENT GUARD
# ----------------------------------------------------------------------
"""
    NonfiniteMomentError(msg)

Raised when a GIV moment evaluation produces a nonfinite (`NaN`/`Inf`) value.
Carries a named diagnosis of the offending entities and/or moments so callers see
the root channel instead of NLsolve's opaque `IsFiniteException` ("the evaluation
of the following equation(s) resulted in a non-finite number: …").
"""
struct NonfiniteMomentError <: Exception
    msg::String
end
Base.showerror(io::IO, e::NonfiniteMomentError) = print(io, "NonfiniteMomentError: ", e.msg)

# Aggregate a per-observation nonfinite mask up to the entity indices it touches.
function _entities_with_nonfinite(u, obs_index)
    bad = falses(obs_index.N)
    @inbounds for idx in eachindex(u)
        if !isfinite(u[idx])
            bad[obs_index.ids[idx]] = true
        end
    end
    return findall(bad)
end

# Compact preview of an index vector for error messages: first `k`, then a count.
_preview(v; k=10) = length(v) <= k ? string(v) : "$(v[1:k]) … ($(length(v)) total)"

"""
    diagnose_nonfinite_moment(u, prec, weightsum, obs_index) -> NonfiniteMomentError

Inspect the `:iv` moment-pipeline intermediates and name the earliest channel that
drives a nonfinite moment, so a solver failure reports *why* instead of surfacing
NLsolve's opaque `IsFiniteException`. Channels, in dependency order:

 1. nonfinite residuals `u` — upstream `NaN`/`Inf` in the response or regressors
    (e.g. a `0/0` flow from a zero-size entity); a single nonfinite response
    observation poisons the shared OLS-FE residualization, so `u` (hence every
    downstream quantity) goes nonfinite for *all* entities.
 2. degenerate entity precision `prec = 1/σ²` — near-zero residual variance ⇒ `Inf`
    precision, or `NaN` propagated from channel 1.
 3. zero/nonfinite per-moment weight `momweight = Σₜ|weightsum|` — a moment with
    zero total weight divides its error row by `0`; an `Inf` weight poisons the
    shared normalization `momweight ./= sum(momweight)`, turning every row nonfinite.

Returns a `NonfiniteMomentError` naming the offending entity / moment indices
(entity indices are in the package's sorted-id order; map them back to your panel's
id labels externally).
"""
function diagnose_nonfinite_moment(u, prec, weightsum, obs_index)
    lines = String[]

    bad_u = _entities_with_nonfinite(u, obs_index)
    if !isempty(bad_u)
        push!(lines, "residuals u nonfinite for $(length(bad_u))/$(obs_index.N) entities " *
            "(entity indices $(_preview(bad_u))); check the response and regressors for NaN/Inf " *
            "before estimation (a common cause is a 0/0 flow from a zero-size entity).")
    end

    bad_prec = findall(!isfinite, prec)
    if !isempty(bad_prec)
        push!(lines, "entity precision 1/σ² nonfinite for $(length(bad_prec)) entities " *
            "(indices $(_preview(bad_prec))); near-zero residual variance ⇒ Inf precision, " *
            "or NaN propagated from nonfinite residuals.")
    end

    momweight_raw = vec(sum(abs.(weightsum); dims=2))
    bad_mw = findall(m -> !isfinite(m) || iszero(m), momweight_raw)
    if !isempty(bad_mw)
        push!(lines, "per-moment total weight zero or nonfinite for moments $(_preview(bad_mw)); " *
            "zero-weight moments divide the error row by 0 and Inf weights poison the shared " *
            "normalization so every moment row becomes nonfinite.")
    end

    isempty(lines) && push!(lines, "a moment evaluation produced a nonfinite value but no " *
        "entity/moment channel was isolated; inspect the panel manually.")

    return NonfiniteMomentError(join(lines, "\n  "))
end

"""
    diagnose_nonfinite_at_guess(guess, q, Cp, C, S, obs_index; precision=nothing)

Recompute the moment-pipeline intermediates at `guess` — residuals `u`, entity
precision `prec` (CUE `1/σ²` unless fixed `precision` supplied), and the O(N)
`weightsum` — and return the named `NonfiniteMomentError`. Called once by
`estimate_giv` when the *initial* moment evaluation is nonfinite (the point at
which NLsolve would otherwise raise its opaque `IsFiniteException`), so the cost
is paid only on the failing solve. The `u`/`prec` channels are exact for every
algorithm; the `weightsum` (momweight) channel uses the `:iv` O(N) form.
"""
function diagnose_nonfinite_at_guess(guess, q, Cp, C, S, obs_index; precision=nothing)
    u = q .+ Cp * guess
    prec = isnothing(precision) ? (1 ./ calculate_entity_variance(u, obs_index)) : precision
    Nm, T, N = length(guess), obs_index.T, obs_index.N
    weightsum = zeros(eltype(u), Nm, T)
    err = zeros(eltype(u), Nm, T)
    fast_pass!(weightsum, err, u, C, S, prec, obs_index, Matrix{eltype(u)}(undef, N, 0), 0)
    return diagnose_nonfinite_moment(u, prec, weightsum, obs_index)
end
