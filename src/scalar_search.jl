function transform_matricies_for_scalar_search(Cpts, Cts, Smat)
    # check algorithm compatibility
    if norm(var(Smat; dims = 2)) > eps(eltype(Smat)) || any(≉(0), var(Cts; dims = 2))
        throw(ArgumentError("For the scalar-search algorithm, weights&coefficients have to be constant across time."))
    end
    coefmapping = Cts[:, 1, :]
    S = Smat[:, 1]
    if !isa(coefmapping, BitArray) || any(!=(1), sum(coefmapping; dims = 2))
        throw(ArgumentError("For the scalar-search algorithm, each id has to have exactly one coefficient on the endogenous variable"))
    end
    p = sum(Cpts; dims = 3)[1:1, :, 1]
    return p, S, coefmapping
end

function estimate_giv(
    q,
    Cp,
    C,
    S,
    obs_index,
    ::Val{:scalar_search};
    guess=nothing,
    constraints=nothing,
    quiet = false,
    complete_coverage=true,
    solver_options=(; ftol=1e-6),
    n_pcs=0,
    kwargs...,
)
    tol = solver_options.ftol
    # Check if panel is balanced before proceeding
    N, T = obs_index.N, obs_index.T
    
    # PC extraction not supported with scalar search
    if n_pcs > 0
        throw(ArgumentError("PC extraction is not supported with the scalar search algorithm. Use :iv, :iv_twopass, or :debiased_ols instead."))
    end

    # Verify that we have a balanced panel (N*T observations)
    if length(q) != N * T
        throw(ArgumentError("Scalar search algorithm requires a balanced panel"))
    end

    # Reshape stacked vectors back to matrices/tensors
    qmat = reshape(q, N, T)

    # For C, reshape from (N*T, K) to (N, T, K)
    Nmom = size(C, 2)
    Cts = BitArray(reshape(C, N, T, Nmom))
    Cpts = reshape(Cp, N, T, Nmom)

    Smat = reshape(S, N, T)

    # Call the original implementation with reshaped matrices
    p, S_vec, coefmapping = transform_matricies_for_scalar_search(Cpts, Cts, Smat)

    if length(guess) == 1
        ζSvec = [find_zero(x -> ζS_err(x, qmat, p, S_vec, coefmapping; kwargs...)[2], guess[1])]
    else
        ζSvec = find_zeros(x -> ζS_err(x, qmat, p, S_vec, coefmapping; kwargs...)[2], guess...)
    end

    abserr = [
        ζS_err(ζ, qmat, p, S_vec, coefmapping; minimizer_backup=false, kwargs...)[2]^2 for ζ in ζSvec
    ]

    if count(<(tol), abserr) > 0
        if count(<(tol), abserr) > 1
            @warn "Multiple solutions found."
        end
        ζS = ζSvec[findfirst(<(tol), abserr)]
        converged = true
    else
        if !quiet
            @warn "No exactly matched ζS found. A minimizer is used."
        end
        converged = false
        ζS = ζSvec[argmin(abserr)]
    end

    ζ̂vecs = solve_ζi(ζS, qmat, p, S_vec, coefmapping; kwargs...)
    ζ̂, err = pick_closest_ζ(ζ̂vecs, ζS, S_vec, coefmapping)

    return ζ̂, converged
end

# function scalar_search_error(ζS, df, formula, id, t, weight; kwargs...)
#     df = preprocess_dataframe(df, formula, id, t, weight)
#     qmat, pmat, Cts, ηts, Smat, uqmat, λq, uCpts, λCp =
#         generate_matrices(df, formula, id, t, weight; algorithm = :scalar_search)
#     p, S, coefmapping = transform_matricies_for_scalar_search(uCpts, Cts, Smat)
#     err = ζS_err.(ζS, Ref(qmat), Ref(p), Ref(S), Ref(coefmapping); kwargs...)
#     return err
# end

function ζS_err(ζS, qmat, p, S, coefmapping; kwargs...)
    ζvecs = solve_ζi(ζS, qmat, p, S, coefmapping; kwargs...)
    ζ, err = pick_closest_ζ(ζvecs, ζS, S, coefmapping)
    return ζ, err
end

function pick_closest_ζ(ζvecs, ζS, S, coefmapping = I)
    ζcomb = map(collect, Iterators.product(ζvecs...))
    err = [S' * coefmapping * ζ - ζS for ζ in ζcomb]
    return ζcomb[argmin(abs.(err))], err[argmin(abs.(err))]
end

function ζi_err(ζi, q, p, ζS, S)
    T = size(q)[2]
    σ² = compute_σ²(q, p, ζi)
    Σqp = vec(mean(q .* p; dims = 2) * T / (T - 1))
    σp² = sum(p .^ 2) / (T - 1)
    numeratorvec = Σqp - 1 / ζS * σ² .* S
    numerator = mean(numeratorvec, weights(1 ./ σ²))
    return ζi + numerator / σp²
end

compute_σ²(q, p, ζ) = var.(eachrow(q .+ ζ .* p), mean = 0.0)

function solve_ζi(ζS, qmat, p, S, coefmapping; minimizer_backup = true)
    Ncat = size(coefmapping)[2]
    ζ = Vector{Vector{Float64}}(undef, Ncat)
    constraint = [-2 * (ζS - 1), 2 * (ζS + 1)]

    for icat in 1:Ncat
        inds = coefmapping[:, icat]
        ζ[icat] = find_zeros(x -> ζi_err(x, qmat[inds, :], p, ζS, S[inds]), constraint)
        if length(ζ[icat]) == 0
            if !minimizer_backup
                ζ[icat] = [NaN]
            else
                ζ[icat] =
                    optimize(x -> ζi_err(x[1], qmat[inds, :], p, ζS, S[inds])^2, [ζS / 2]).minimizer
            end
        end
    end
    return ζ
end
