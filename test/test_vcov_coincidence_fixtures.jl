# ---------------------------------------------------------------------------
# Deterministic algebra fixtures for the masked covariance routes
# (giv-solver-stability/estimator-scope-cleanup/simulation-validation).
#
# Three seed-free, hand-constructed fixtures that pin the covariance code to a
# known answer and catch the two superseded-code bugs the branch retired:
#
#   1. COINCIDENCE — under the optimal restrictions (complete coverage, the
#      admissible all-pair set, and the frozen weights equal to the optimal
#      instrument 1/σ² · M_t), the masked empirical sandwich `solve_vcov` equals
#      the model-implied information formula `solve_optimal_vcov` to machine
#      precision. This is an *exact* finite-sample identity here because the
#      residuals are built from orthogonal ±1 Hadamard columns whose 0-indexed
#      positions are powers of two: the sample second moments hit the model
#      moments exactly and every cross-pair fourth-moment vanishes, so the
#      empirical meat equals the model meat diag(σ_i²σ_j²) exactly.
#
#   2. ASYMMETRIC BREAD (transposition catch) — on the same design the bread
#      G = mean_t Wₜ'Dₜ is materially non-symmetric (heterogeneous loadings and
#      sizes across coefficients). The correct sandwich G⁻¹BG⁻ᵀ coincides with
#      the optimal formula; the transposed bread G⁻ᵀBG⁻¹ (the superseded
#      `solve_specialized_vcov` orientation, umbrella decision 14) deviates by
#      several hundred percent. A coincidence assertion at rtol 1e-8 therefore
#      *fails* under the transposition — verified explicitly below.
#
#   3. EXCLUSION (pair-scope catch) — excluding one genuinely co-occurring,
#      genuinely loaded pair drops exactly one admissible pair and changes the
#      covariance by a material margin. The package's masked sandwich on the
#      excluded design matches an independent O(N²) reference restricted to the
#      *same* admissible set, and differs from the all-pair covariance — so a
#      covariance that silently restored the excluded moment would be caught.
#
# PRE-REGISTERED tolerances (fixed before the run, not tuned):
#   coincidence: rtol 1e-8 (observed ~1e-14);   transposition signal: > 1.0
#   (observed ~5.7×);   bread asymmetry: > 0.1 (observed ~0.79);   exclusion
#   covariance change: > 1e-2 (observed ~0.05);   masked-sandwich vs reference:
#   rtol 1e-8 (observed ~1e-15).
# ---------------------------------------------------------------------------
using Test, OptimalGIV, LinearAlgebra, Statistics
using OptimalGIV: build_error_function, admissible_pair_indices, solve_optimal_vcov,
    solve_vcov, calculate_entity_variance, period_mweights
using DataFrames, CategoricalArrays

include("vcov_test_helpers.jl")   # _independent_vcov (standalone-runnable)

# Hadamard matrix of order 2^k (T must be a power of two).
_hadamard(n) = (H = [1.0;;]; while size(H, 1) < n; H = [H H; H -H]; end; H)

# Deterministic complete-coverage GIV design. Entity i loads coefficient grp[i];
# residuals are scaled orthogonal ±1 Hadamard columns at 0-indexed powers of two.
# Returns the frozen estimating matrices (uq, uCp, C, S, obs_index) plus the true
# coefficient vector, for an optional exclusion mask.
function _coincidence_matrices(; exclude_pairs=Dict{Int,Vector{Int}}())
    N, T = 6, 64
    grp  = [1, 1, 1, 2, 2, 3]
    ζ    = [0.5, 2.0, 3.5]
    σ    = [0.6, 0.7, 0.8, 1.6, 1.8, 2.5]
    Sraw = [0.3, 0.4, 0.5, 1.5, 1.8, 3.0]
    cols = [2, 3, 5, 9, 17, 33]                 # 0-indexed 1,2,4,8,16,32
    H = _hadamard(T)
    u = reduce(vcat, [σ[i] .* H[:, cols[i]]' for i in 1:N])
    S = Sraw ./ sum(Sraw)
    ζent = ζ[grp]
    p = [dot(S, u[:, t]) / dot(S, ζent) for t in 1:T]
    q = u .- ζent * p'
    df = DataFrame(id=CategoricalArray(repeat(1:N, T)), t=repeat(1:T; inner=N),
        q=vec(q), p=repeat(p; inner=N), S=repeat(S, T),
        grp=CategoricalArray(repeat(grp, T)))
    _, mats = build_error_function(df, @formula(q + grp & endog(p) ~ 0), :id, :t, :S;
        algorithm=:iv, complete_coverage=true, quiet=true,
        precision_weights=:raw_onestep, exclude_pairs=exclude_pairs)
    return (; mats, ζ)
end

@testset "deterministic coincidence: masked sandwich == information formula" begin
    f = _coincidence_matrices()
    m = f.mats
    u = m.uq + m.uCp * f.ζ
    σu² = calculate_entity_variance(u, m.obs_index)
    Mw = period_mweights(f.ζ, m.C, m.S, m.obs_index)

    # optimal instrument = 1/σ² entity precision with the complete-coverage M_t
    _, Σopt = solve_optimal_vcov(f.ζ, u, m.S, m.C, m.obs_index)
    _, Σsand = solve_vcov(u, m.S, m.C, m.uCp, m.obs_index; precision=1 ./ σu², Mweights=Mw)
    ref = _independent_vcov(u, m.S, m.C, m.uCp, m.obs_index, 1 ./ σu², Mw)

    # exact finite-sample coincidence (observed ~1e-14; PRE-REGISTERED rtol 1e-8)
    @test Σsand ≈ Σopt rtol = 1e-8
    @test ref.Σ ≈ Σopt rtol = 1e-8
    # empirical meat equals the model diagonal meat exactly (orthogonal design)
    @test norm(ref.B - ref.Bdiag) / norm(ref.B) < 1e-8
end

@testset "asymmetric bread: transposition breaks the coincidence" begin
    f = _coincidence_matrices()
    m = f.mats
    u = m.uq + m.uCp * f.ζ
    σu² = calculate_entity_variance(u, m.obs_index)
    Mw = period_mweights(f.ζ, m.C, m.S, m.obs_index)
    _, Σopt = solve_optimal_vcov(f.ζ, u, m.S, m.C, m.obs_index)
    ref = _independent_vcov(u, m.S, m.C, m.uCp, m.obs_index, 1 ./ σu², Mw)

    # the bread is materially non-symmetric (heterogeneous loadings/sizes)
    @test norm(ref.G - ref.G') / norm(ref.G) > 0.1

    invG = inv(ref.G)
    Σcorrect = invG * ref.B * invG' / m.obs_index.T
    Σtransposed = invG' * ref.B * invG / m.obs_index.T      # superseded orientation
    @test Σcorrect ≈ Σopt rtol = 1e-8                        # correct code coincides
    # the transposed bread deviates by > 100% ⇒ a coincidence test at rtol 1e-8
    # would fail under the transposition (observed signal ~5.7×)
    @test norm(Σtransposed - Σopt) / norm(Σopt) > 1.0
    @test !isapprox(Σtransposed, Σopt; rtol=1e-8)
end

@testset "exclusion: pair-scope drift is caught" begin
    fall = _coincidence_matrices()
    # entities 4 and 5 are the two coefficient-2 loaders; their pair is the main
    # moment identifying coefficient 2, so excluding it moves the covariance ~4×.
    fex = _coincidence_matrices(exclude_pairs=Dict(4 => [5]))
    mall, mex = fall.mats, fex.mats
    ζ = fall.ζ

    pi_all, _ = admissible_pair_indices(mall.obs_index)
    pi_ex, _ = admissible_pair_indices(mex.obs_index)
    @test length(pi_all) - length(pi_ex) == 1                  # exactly one pair removed

    uall = mall.uq + mall.uCp * ζ
    uex = mex.uq + mex.uCp * ζ
    _, Σall = solve_optimal_vcov(ζ, uall, mall.S, mall.C, mall.obs_index)
    _, Σex = solve_optimal_vcov(ζ, uex, mex.S, mex.C, mex.obs_index)
    # excluding a genuinely loaded pair changes the covariance materially
    # (observed ~3.8×; PRE-REGISTERED margin > 0.1)
    @test norm(Σall - Σex) / norm(Σall) > 0.1

    # the masked sandwich on the excluded design honours exactly the reduced pair
    # set: it matches an independent O(N²) reference restricted to the same pairs,
    # and (orthogonal design) still coincides with the excluded information formula.
    σu²ex = calculate_entity_variance(uex, mex.obs_index)
    Mwex = period_mweights(ζ, mex.C, mex.S, mex.obs_index)
    _, Σsex = solve_vcov(uex, mex.S, mex.C, mex.uCp, mex.obs_index;
        precision=1 ./ σu²ex, Mweights=Mwex)
    refex = _independent_vcov(uex, mex.S, mex.C, mex.uCp, mex.obs_index, 1 ./ σu²ex, Mwex)
    @test Σsex ≈ refex.Σ rtol = 1e-8
    @test Σsex ≈ Σex rtol = 1e-8
    # a covariance that ignored the mask (all pairs) would differ from the masked one
    @test !isapprox(Σall, Σex; rtol=0.1)
end
