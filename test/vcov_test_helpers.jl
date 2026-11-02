# ---------------------------------------------------------------------------
# Shared fixtures and the independent O(N²) sandwich reference used across
# test_fixed_weight_mode.jl / test_vcov_scope.jl / test_analytic_jacobian.jl.
#
# Loaded via `include("vcov_test_helpers.jl")` at the top of each of those files.
# The `@isdefined` guard makes repeated includes (runtests.jl includes all three
# in sequence) a no-op, and lets each file be run standalone.
# ---------------------------------------------------------------------------
using OptimalGIV, Random, LinearAlgebra, Statistics
using DataFrames, CSV, CategoricalArrays

if !@isdefined(VCOV_TEST_HELPERS_LOADED)
    const VCOV_TEST_HELPERS_LOADED = true

    const _SIMDATA1 = joinpath(@__DIR__, "..", "examples", "simdata1.csv")

    function _load_simdata1()
        df = CSV.read(_SIMDATA1, DataFrame)
        df.id = CategoricalArray(df.id)
        return df
    end

    # numerical equivalence between :iv and :iv_twopass only holds when the endogenous
    # variable is excluded from the instrument (fe(id) & (η1+η2), no id&η)
    const _FEQ = @formula(q + id & endog(p) ~ fe(id) & (η1 + η2) + 0)

    # DGP with genuinely time-varying aggregate elasticity ζS_t: sizes S_it move
    # over t and p_t clears the market exactly, so the complete-coverage period
    # multipliers are non-uniform. Returns the frozen-weight test formula alongside.
    function _timevarying_complete_fixture(; N=5, T=80, seed=20260720)
        @assert N == 5 "the frozen ζtrue below is fixed for N=5"
        Random.seed!(seed)
        ζtrue = [0.5, 1.0, 1.5, 2.0, 3.0]
        Smat = rand(N, T) .^ 3 .+ 0.05
        Smat ./= sum(Smat; dims=1)
        umat = randn(N, T) .* (0.5 .+ rand(N))
        p = [dot(Smat[:, t], umat[:, t]) / dot(Smat[:, t], ζtrue) for t in 1:T]
        qmat = umat .- ζtrue * p'
        df = DataFrame(
            id=CategoricalArray(repeat(1:N, T)),
            t=repeat(1:T; inner=N),
            q=vec(qmat),
            p=repeat(p; inner=N),
            S=vec(Smat),
        )
        fml = @formula(q + id & endog(p) ~ 0)
        return (; df, fml, ζtrue)
    end

    # Independent O(N²) construction of the masked empirical sandwich directly from
    # the estimating equation. This deliberately does NOT call the production pair
    # catalog, period-score helper, or score constructor, so it is a genuine cross
    # check of the fast implementation. Returns the covariance plus the intermediate
    # bread/meat pieces (G, B), the pairwise-diagonal meat (Bdiag, dropping cross-pair
    # covariance), and the period scores.
    function _independent_vcov(u, S, C, Cp, obs_index, precision, Mweights)
        K, T = size(C, 2), obs_index.T
        scores = zeros(K, T)
        weightsum = zeros(K, T)
        Gt = zeros(K, K, T)
        pair_scores = [Vector{Vector{Float64}}() for _ in 1:T]
        for t in 1:T
            r = obs_index.start_indices[t]:obs_index.end_indices[t]
            mt = isnothing(Mweights) ? 1.0 : Mweights[t]
            for ii in r, jj in (ii + 1):last(r)
                i, j = obs_index.ids[ii], obs_index.ids[jj]
                obs_index.exclpairs[i, j] && continue
                w = [precision[i] * S[jj] * C[ii, k] +
                     precision[j] * S[ii] * C[jj, k] for k in 1:K]
                d = [Cp[ii, k] * u[jj] + u[ii] * Cp[jj, k] for k in 1:K]
                h = u[ii] * u[jj]
                scores[:, t] .+= mt .* w .* h
                weightsum[:, t] .+= w
                Gt[:, :, t] .+= mt .* (w * d')
                push!(pair_scores[t], mt .* w .* h)
            end
        end
        momweight = vec(sum(abs.(weightsum); dims=2))
        momweight ./= sum(momweight)
        scores ./= reshape(momweight, :, 1)
        Gt ./= reshape(momweight, :, 1, 1)
        G = dropdims(mean(Gt; dims=3); dims=3)
        centered_scores = scores .- mean(scores; dims=2)
        B = centered_scores * centered_scores' / T
        Bdiag = zeros(K, K)
        for pair_idx in eachindex(pair_scores[1])
            pair_series = hcat([pair_scores[t][pair_idx] ./ momweight for t in 1:T]...)
            pair_series .-= mean(pair_series; dims=2)
            Bdiag .+= pair_series * pair_series' / T
        end
        invG = inv(G)
        Σ = invG * B * invG' / T
        return (; Σ=Matrix(Symmetric(Σ)), G, B, Bdiag, scores)
    end
end
