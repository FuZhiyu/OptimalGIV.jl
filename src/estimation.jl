function estimate_giv(
    q,
    Cp,
    C,
    S,
    obs_index,
    ::A;
    guess=nothing,
    quiet=false,
    complete_coverage=true,
    solver_options=(;),
) where {A<:Union{Val{:iv},Val{:iv_twopass},Val{:debiased_ols}}}
    if isnothing(guess)
        if !quiet
            @info "Initial guess is not provided. Using OLS estimate as initial guess."
        end
        guess = (Cp' * Cp) \ (Cp' * q)
    end

    Nmom = size(Cp, 2)
    err0 = mean_moment_conditions(guess, q, Cp, C, S, obs_index, complete_coverage, A())
    if length(err0) != Nmom
        throw(ArgumentError("The number of moment conditions is not equal to the number of initial guess."))
    end

    res = nlsolve(
        x -> mean_moment_conditions(x, q, Cp, C, S, obs_index, complete_coverage, A()),
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

function moment_conditions(ζ, q, Cp, C, S, obs_index, complete_coverage, ::Val{:iv_twopass})
    Nmom = length(ζ)
    N, T = obs_index.N, obs_index.T


    # Calculate residuals
    u = q + Cp * ζ

    # Calculate variance by entity
    σu²vec = calculate_entity_variance(u, obs_index)
    precision = 1 ./ σu²vec
    # precision = precision ./ sum(precision)

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

                weight_i = C[idx_i, imom] * precision[i]

                for idx_j in (idx_i+1):end_idx
                    j = obs_index.ids[idx_j]

                    # Since idx_i < idx_j within a time period, we know i ≠ j
                    if obs_index.exclpairs[i, j]
                        continue
                    end

                    err[imom, t] += u[idx_i] * u[idx_j] * weight_i * S[idx_j]
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

                weight_j = C[idx_j, imom] * precision[j]

                for idx_i in start_idx:(idx_j-1)
                    i = obs_index.ids[idx_i]

                    # Since idx_i < idx_j within a time period, we know i ≠ j
                    if obs_index.exclpairs[i, j]
                        continue
                    end

                    err[imom, t] += u[idx_i] * u[idx_j] * weight_j * S[idx_i]
                    weightsum[imom, t] += weight_j * S[idx_i]
                end

            end
        end
    end

    # the efficient weighting requires the scaling of multiplier for each period
    # only feasible when we observe the full market
    if complete_coverage
        ζS = solve_aggregate_elasticity(ζ, C, S, obs_index)
        Mweights = 1 ./ clamp.(abs.(ζS), sqrt(eps(eltype(ζS))), Inf) # avoid division by zero
        Mweights ./= sum(Mweights)
        err .*= Mweights'
        # weightsum .*= Mweights'
    end

    # equal weight the final moment conditions for numerical stability
    # it's exactly identified at this point hence it's not affecting the estimation
    momweight = sum(abs.(weightsum); dims=2)
    momweight ./= sum(momweight)
    err ./= momweight

    return err
end


function moment_conditions(ζ, q, Cp, C, S, obs_index, complete_coverage, ::Val{:iv})
    Nm = length(ζ)
    T = obs_index.T
    # Compute period weights if complete_coverage constraint holds
    Mweights = ones(eltype(ζ), T)
    if complete_coverage
        ζSvec = solve_aggregate_elasticity(ζ, C, S, obs_index)
        Mvec = 1 ./ ζSvec
        Mweights .= Mvec ./ sum(Mvec)
    end
    err = zeros(eltype(ζ), Nm, T)

    # residuals and entity-level precision ------------------------------
    u = q .+ Cp * ζ
    σu²vec = calculate_entity_variance(u, obs_index)
    prec = inv.(σu²vec)

    weightsum = zeros(eltype(ζ), Nm, T)
    # 1️⃣ fast O(N) pass
    fast_pass!(weightsum, err, u, C, S, prec, obs_index)

    # 2️⃣ subtract excluded pairs
    deduct_excluded_pairs!(err, weightsum, C, S, u, prec, obs_index)

    # the efficient weighting requires the scaling of multiplier for each period
    # only feasible when we observe the full market
    if complete_coverage
        ζS = solve_aggregate_elasticity(ζ, C, S, obs_index)
        Mweights = 1 ./ clamp.(abs.(ζS), sqrt(eps(eltype(ζS))), Inf) # avoid division by zero
        Mweights ./= sum(Mweights)
        err .*= Mweights'
        # weightsum .*= Mweights'
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
function fast_pass!(weightsum, err, u, C, S, prec, obs_index)
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
function deduct_excluded_pairs!(err, weightsum, C, S, u, prec, obs_index)
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

                    # Deduct contribution for excluded pairs
                    err[imom, t] -= u[idx_i] * u[idx_j] * weight_i * S[idx_j]
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

                    # Deduct contribution for excluded pairs
                    err[imom, t] -= u[idx_i] * u[idx_j] * weight_j * S[idx_i]
                    weightsum[imom, t] -= weight_j * S[idx_i]
                end
            end
        end
    end

    return nothing
end


function moment_conditions(ζ, q, Cp, C, S, obs_index, complete_coverage, ::Val{:debiased_ols})
    Nmom = length(ζ)
    N, T = obs_index.N, obs_index.T

    # Calculate residuals
    u = q + Cp * ζ

    # Calculate variance by entity
    σu²vec = calculate_entity_variance(u, obs_index)
    precision = 1 ./ σu²vec
    precision ./= sum(precision)

    # Initialize error and weightsum
    err = zeros(eltype(ζ), Nmom, T)
    weightsum = zeros(eltype(ζ), Nmom, T)

    # Calculate ζS for each time period
    ζSvec = solve_aggregate_elasticity(ζ, C, S, obs_index)

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

                weight = precision[i]
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

mean_moment_conditions(ζ, args...) = vec(mean(moment_conditions(ζ, args...); dims=2))

function estimate_loading_on_η(x, η, cholη=cholesky(η' * η))
    λ = cholη \ (η' * x)
    return λ
end

function residualize_on_η(x, η, cholη=cholesky(η' * η))
    λ = estimate_loading_on_η(x, η, cholη)
    return x - η * λ, λ
end

function solve_aggregate_elasticity(ζ, C, S, obs_index; complete_coverage=true)
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

function solve_optimal_vcov(ζ, u, S, C, obs_index)
    Nmom = length(ζ)
    N, T = obs_index.N, obs_index.T
    σu²vec = calculate_entity_variance(u, obs_index)
    ζSvec = solve_aggregate_elasticity(ζ, C, S, obs_index)
    Mvec = 1 ./ ζSvec

    # Step 1: Efficiently identify all unique entity pairs that co-occur
    # 1) Build presence‐matrix
    M = obs_index.entity_obs_indices .> 0   # N×T BitMatrix
    # 2) Compute co‐occurrence counts
    co = M * transpose(M)                   # N×N dense Int matrix
    # 3) findall gives you CartesianIndex's for each true entry
    inds = findall(triu(co .> zero(eltype(co)), 1))       # CartesianIndices of every non-zero count in the upper triangle
    pair_i = [I[1] for I in inds]
    pair_j = [I[2] for I in inds]
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



function solve_vcov(u, S, C, Cp, obs_index)
    Nmom = size(C, 2)
    N, T = obs_index.N, obs_index.T
    σu²vec = calculate_entity_variance(u, obs_index)

    # Step 1: Efficiently identify all unique entity pairs that co-occur
    # 1) Build presence‐matrix
    M = obs_index.entity_obs_indices .> 0   # N×T BitMatrix
    # 2) Compute co‐occurrence counts
    co = M * transpose(M)                   # N×N dense Int matrix
    # 3) findall gives you CartesianIndex's for each true entry
    inds = findall(triu(co .> zero(eltype(co)), 1))       # CartesianIndices of every non-zero count in the upper triangle
    pair_i = [I[1] for I in inds]
    pair_j = [I[2] for I in inds]
    # Number of unique entity pairs
    n_pairs = length(pair_i)

    # Step 2: Initialize arrays for computation
    Vdiag = zeros(n_pairs)
    W = zeros(n_pairs, Nmom, T)
    D = zeros(n_pairs, Nmom, T)

    # Step 3: Compute Vdiag for all pairs
    for idx in 1:n_pairs
        i, j = pair_i[idx], pair_j[idx]
        Vdiag[idx] = σu²vec[i] * σu²vec[j]
    end

    precision = 1 ./ σu²vec
    # Step 4: Compute W matrix (previous D without Mvec scaling)
    for t in 1:T
        for idx in 1:n_pairs
            i, j = pair_i[idx], pair_j[idx]
            i_pos = obs_index.entity_obs_indices[i, t]
            j_pos = obs_index.entity_obs_indices[j, t]
            if i_pos == 0 || j_pos == 0
                continue
            end
            for k in 1:Nmom
                # when we do not have the full market, we do not scale it by Mvec
                W[idx, k, t] = precision[i] * S[j_pos] * C[i_pos, k] + precision[j] * S[i_pos] * C[j_pos, k]
                D[idx, k, t] = u[j_pos] * Cp[i_pos, k] + u[i_pos] * Cp[j_pos, k]
            end
        end
    end

    # Step 6: Final calculation via sandwich formula
    # Compute sums of D'W and W'Vdiag W across time periods
    A = zeros(eltype(u), Nmom, Nmom)
    B = zeros(eltype(u), Nmom, Nmom)
    @views for t in 1:T
        Wt = W[:, :, t]
        Dt = D[:, :, t]

        A .+= Dt' * Wt
        B .+= Wt' * (Wt .* Vdiag)
    end
    A ./= (T - 1) # to match the solve_vcov as the σu²vec is scaled by T-1
    B ./= T
    B = Symmetric(B + B') / 2
    # Sandwich variance
    invA = inv(A)
    Σζ = invA * B * invA' / T
    Σζ = Symmetric(Σζ + Σζ') / 2
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

    # Preallocate the final matrices
    bread = zeros(Nmom, Nmom)
    meat = zeros(Nmom, Nmom)

    # Process a chunk of entities
    function process_chunk(chunk_range)::Tuple{Matrix{Float64},Matrix{Float64}}
        local_bread = zeros(Nmom, Nmom)
        local_meat = zeros(Nmom, Nmom)

        for i in chunk_range
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

            # Update local accumulators
            local_bread .+= full_ηη_i
            local_meat .+= full_ηη_i .* σu²vec[i]
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