function estimate_giv(
    q,
    Cp,
    C,
    S,
    exclmat,
    obs_index,
    ::A;
    guess = nothing,
    quiet = false,
    solver_options = (;),
) where {A<:Union{Val{:iv},Val{:iv_legacy},Val{:debiased_ols}}}
    if isnothing(guess)
        if !quiet
            @info "Initial guess is not provided. Using OLS estimate as initial guess."
        end
        guess = (Cp' * Cp) \ (Cp' * q)
    end

    Nmom = size(Cp, 2)
    err0 = mean_moment_conditions(guess, q, Cp, C, S, exclmat, obs_index, A())
    if length(err0) != Nmom
        throw(ArgumentError("The number of moment conditions is not equal to the number of initial guess."))
    end

    res = nlsolve(
        x -> mean_moment_conditions(x, q, Cp, C, S, exclmat, obs_index, A()),
        guess;
        solver_options...,
    )

    converged = res.f_converged
    ζ̂ = res.zero

    if !converged && !quiet
        @warn "The estimation did not converge."
    end

    return ζ̂, converged
end

function moment_conditions(ζ, q, Cp, C, S, exclmat, obs_index, ::Val{:iv_legacy})
    Nmom = length(ζ)
    N, T = obs_index.N, obs_index.T

    # Calculate residuals
    u = q + Cp * ζ

    # Calculate variance by entity
    σu²vec = calculate_entity_variance(u, obs_index)
    precision = 1 ./ σu²vec
    precision = precision ./ sum(precision)

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
            for idx_i in start_idx:end_idx
                i = obs_index.ids[idx_i]

                # Skip if C is zero
                if iszero(C[idx_i, imom])
                    continue
                end

                weight_i = C[idx_i, imom] * precision[i]

                for idx_j in (idx_i+1):end_idx
                    j = obs_index.ids[idx_j]

                    # Since idx_i < idx_j within a time period, we know i ≠ j
                    if exclmat[i, j]
                        continue
                    end

                    err[imom, t] += u[idx_i] * u[idx_j] * weight_i * S[idx_j]
                end

                weightsum[imom, t] += precision[i]
            end

            # Second loop: iterate over j, then i < j
            for idx_j in start_idx:end_idx
                j = obs_index.ids[idx_j]

                # Skip if C is zero
                if iszero(C[idx_j, imom])
                    continue
                end

                weight_j = C[idx_j, imom] * precision[j]

                for idx_i in start_idx:(idx_j-1)
                    i = obs_index.ids[idx_i]

                    # Since idx_i < idx_j within a time period, we know i ≠ j
                    if exclmat[i, j]
                        continue
                    end

                    err[imom, t] += u[idx_i] * u[idx_j] * weight_j * S[idx_i]
                end

                weightsum[imom, t] += precision[j]
            end
        end
    end

    # Normalize
    # momweight = sum(abs.(weightsum); dims=2)
    # momweight = momweight ./ sum(momweight)
    # err ./= momweight

    return err
end


function moment_conditions(ζ, q, Cp, C, S, exclmat, obs_index, ::Val{:iv})

    Nm = length(ζ)
    T = obs_index.T
    err = zeros(eltype(ζ), Nm, T)

    # residuals and entity-level precision ------------------------------
    u = q .+ Cp * ζ
    σu²vec = calculate_entity_variance(u, obs_index)
    prec = inv.(σu²vec)
    prec ./= sum(prec)

    # 1️⃣ fast O(N) pass
    fast_pass!(err, u, C, S, prec, obs_index)

    # 2️⃣ subtract excluded pairs
    deduct_excluded_pairs!(err, C, S, u, prec, exclmat, obs_index)

    return err
end

# ----------------------------------------------------------------------
#  FAST  O(N)  PASS   (fills only `err`)
# ----------------------------------------------------------------------
function fast_pass!(err, u, C, S, prec, obs_index)
    Nm = size(err, 1)
    T = obs_index.T

    @inbounds for t in 1:T
        r = obs_index.start_indices[t]:obs_index.end_indices[t]
        ids_t = obs_index.ids[r]

        u_t = @view u[r]
        S_t = @view S[r]
        prec_t = prec[ids_t]

        b_total = sum(S_t .* u_t)                 # ∑ b_i   (no mask!)

        for m in 1:Nm
            C_t = @view C[r, m]
            nz = C_t .!= 0
            if !any(nz)
                continue
            end

            a_vec = C_t[nz] .* prec_t[nz] .* u_t[nz]           # a_i
            diag_ab = a_vec .* (S_t[nz] .* u_t[nz])              # a_i b_i

            err[m, t] = (sum(a_vec) * b_total - sum(diag_ab))
        end
    end
    return nothing
end


# ----------------------------------------------------------------------
#  SECOND PASS  – deduct excluded (err only, weightsum gone)
# ----------------------------------------------------------------------
function deduct_excluded_pairs!(err, C, S, u, prec, exclmat, obs_index)

    Nm = size(err, 1)
    T = obs_index.T

    @inbounds for t in 1:T
        r = obs_index.start_indices[t]:obs_index.end_indices[t]
        ids_t = obs_index.ids[r]

        u_t = @view u[r]
        S_t = @view S[r]

        for m in 1:Nm
            C_t = @view C[r, m]

            # ---------- i before j ------------------------------------
            for idx_i in eachindex(r)
                Ci = C_t[idx_i]
                if Ci == 0
                    continue
                end     # rows with C == 0 never contribute

                i = ids_t[idx_i]
                wi = Ci * prec[i]
                ui = u_t[idx_i]

                for idx_j in (idx_i+1):length(r)
                    j = ids_t[idx_j]
                    if !exclmat[i, j]
                        continue
                    end

                    err[m, t] -= ui * u_t[idx_j] * wi * S_t[idx_j]
                end
            end

            # ---------- j after i -------------------------------------
            for idx_j in eachindex(r)
                Cj = C_t[idx_j]
                if Cj == 0
                    continue
                end

                j = ids_t[idx_j]
                wj = Cj * prec[j]
                uj = u_t[idx_j]

                for idx_i in 1:(idx_j-1)
                    i = ids_t[idx_i]
                    if !exclmat[i, j]
                        continue
                    end

                    err[m, t] -= u_t[idx_i] * uj * wj * S_t[idx_i]
                end
            end
        end
    end
    return nothing
end


function moment_conditions(ζ, qmat, Cpts, Cts, Smat, exclmat, ::Val{:debiased_ols})
    Nmom = length(ζ)
    N, T = size(qmat)
    u = vec(qmat) + reshape(Cpts, N * T, Nmom) * ζ
    umat = reshape(u, N, T)

    σu²vec = [var(umat[i, :]; mean = zero(eltype(umat))) for i in 1:N]
    precision = 1 ./ σu²vec
    precision = precision ./ sum(precision)
    err = zeros(eltype(ζ), Nmom, T)
    ζSvec = solve_aggregate_elasticity(ζ, Cts, Smat, obs_index)
    weightsum = zeros(eltype(ζ), Nmom, T) # equal weight moment conditions for numerical stability
    @threads for (imom, t) in Tuple.(CartesianIndices((Nmom, T)))
        for i in 1:N
            if iszero(Cts[i, t, imom])
                continue
            end
            weight = precision[i]
            uCp = umat[i, t] * Cpts[i, t, imom]
            CSσ² = umat[i, t]^2 * Cts[i, t, imom] * Smat[i, t]
            # alternatively, use estimated variance
            # CSσ² = σu²vec[i,t] *(T-1)/T * Cts[i, t, imom] * Smat[i, t]
            err[imom, t] += weight * (uCp - CSσ² / ζSvec[t])
            weightsum[imom, t] += weight
        end
    end
    momweight = sum(abs.(weightsum); dims = 2)
    momweight = momweight ./ sum(momweight)
    err ./= momweight # equal weight moment conditions for numerical stability
    return err
end

mean_moment_conditions(ζ, args...) = vec(mean(moment_conditions(ζ, args...); dims = 2))

function estimate_loading_on_η(x, η, cholη = cholesky(η' * η))
    λ = cholη \ (η' * x)
    return λ
end

function residualize_on_η(x, η, cholη = cholesky(η' * η))
    λ = estimate_loading_on_η(x, η, cholη)
    return x - η * λ, λ
end

function solve_aggregate_elasticity(ζ, C, S, obs_index)
    Nmom = length(ζ)
    ζSvec = zeros(eltype(ζ), obs_index.T)

    for t in 1:obs_index.T
        start_idx = obs_index.start_indices[t]
        end_idx = obs_index.end_indices[t]

        # Calculate weighted sum for this time period
        sum_weighted = 0.0
        denominator = 0.0

        # Directly iterate through observation indices for this period
        for idx in start_idx:end_idx
            weight = S[idx]

            # Calculate C * ζ for this observation
            C_ζ = 0.0
            for imom in 1:Nmom
                C_ζ += C[idx, imom] * ζ[imom]
            end

            sum_weighted += C_ζ * weight
            denominator += weight
        end

        ζSvec[t] = sum_weighted
    end

    return ζSvec
end

function solve_vcov(ζ, u, S, C, obs_index)
    Nmom = length(ζ)
    N, T = obs_index.N, obs_index.T
    σu²vec = calculate_entity_variance(u, obs_index)
    ζSvec = solve_aggregate_elasticity(ζ, C, S, obs_index)
    Mvec = 1 ./ ζSvec

    # Step 1: Efficiently identify all unique entity pairs that co-occur
    pair_i = Int[]
    pair_j = Int[]

    # Use the entity_obs_indices matrix to find co-occurring pairs
    # Two entities co-occur if they both have non-zero indices in any time period
    for i in 1:N
        for j in (i+1):N
            # Check if entities i and j ever appear in the same time period
            if any((obs_index.entity_obs_indices[i, :] .> 0) .& (obs_index.entity_obs_indices[j, :] .> 0))
                push!(pair_i, i)
                push!(pair_j, j)
            end
        end
    end

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

            # Calculate D values for all moment conditions
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
Solve the OLS covariance matrix for the GIV estimation.

This version uses the observation indexing structure instead of matrices directly.

Parameters:
- σu²vec: Pre-computed entity-specific residual variances
- η: Matrix of control variables
- obs_index: Observation index structure with entity mappings

Returns:
- Covariance matrix for OLS estimator
"""
function solve_ols_vcov(σu²vec, η, obs_index)
    N, T = obs_index.N, obs_index.T
    Nmom = size(η, 2)

    # Initialize per-thread accumulators to avoid data races
    nthreads = Threads.nthreads()
    breads = [zeros(Nmom, Nmom) for _ in 1:nthreads]
    meats = [zeros(Nmom, Nmom) for _ in 1:nthreads]

    @threads for i in 1:N
        threadid = Threads.threadid()



        # Collect η values for this entity
        η_i = zeros(0, Nmom)
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
            η_i = vcat(η_i, transpose(η[idx, :]))
            n_obs += 1
        end

        # Skip if not enough observations
        if n_obs < 2
            continue
        end

        # Identify non-zero columns to exploit sparsity
        zero_cols = vec(all(iszero, η_i; dims=1))
        nonzero_cols = findall(!x -> x, zero_cols)
        if isempty(nonzero_cols)
            continue  # Skip if all columns are zero
        end

        # Extract non-zero columns
        η_i_nonzero = η_i[:, nonzero_cols]

        # Compute ηη'_i matrix
        ηη_i_sub = zeros(length(nonzero_cols), length(nonzero_cols))
        BLAS.syrk!('U', 'T', 1.0 / n_obs, η_i_nonzero, 0.0, ηη_i_sub)

        # Wrap with Symmetric to represent the full symmetric matrix
        ηη_i_sub_sym = Symmetric(ηη_i_sub, :U)

        # Map back to original dimensions
        full_ηη_i = zeros(Nmom, Nmom)
        full_ηη_i[nonzero_cols, nonzero_cols] .= ηη_i_sub_sym

        # Update per-thread accumulators
        breads[threadid] .+= full_ηη_i
        meats[threadid] .+= full_ηη_i .* σu²vec[i]
    end

    # Sum over all threads
    bread = sum(breads)
    meat = sum(meats)

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