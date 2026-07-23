using Test, OptimalGIV, Random, LinearAlgebra, ForwardDiff, Statistics
using OptimalGIV: calculate_entity_variance, create_observation_index,
    masked_period_scores, mean_moment_conditions, moment_conditions,
    period_mweights, solve_optimal_vcov, solve_vcov,
    _central_difference_jacobian, _positive_domain_jacobian, build_error_function,
    aggregate_elasticity_in_domain
using DataFrames, CategoricalArrays

function _vcov_scope_fixture(; excluded=false)
    Random.seed!(20260722)
    N, T, K = 4, 7, 2
    panel = DataFrame(
        id=CategoricalArray(repeat(1:N, T)),
        t=repeat(1:T; inner=N),
    )
    exclusions = excluded ? Dict(1 => [2]) : Dict{Int,Vector{Int}}()
    obs_index = create_observation_index(panel, :id, :t, exclusions)
    u = randn(N * T)
    Cp = randn(N * T, K)
    C = 0.2 .+ rand(N * T, K)
    S = 0.1 .+ rand(N * T)
    precision = 0.5 .+ rand(N)
    Mweights = collect(1.0:T)
    Mweights ./= sum(Mweights)
    return (; u, Cp, C, S, precision, Mweights, obs_index)
end

function _near_boundary_cue_fixture()
    Random.seed!(20260723)
    N, T, K = 3, 4, 2
    ζ = [1.0, -0.9999999]
    target_aggregate = [1e-7, 1.0, 0.50000005, 0.20000008]
    second_loading = (1 .- target_aggregate) ./ 0.9999999
    panel = DataFrame(
        id=CategoricalArray(repeat(1:N, T)),
        t=repeat(1:T; inner=N),
    )
    obs_index = create_observation_index(panel, :id, :t)
    C = reduce(vcat, [repeat(reshape([1.0, second_loading[t]], 1, K), N, 1) for t in 1:T])
    S = fill(1 / N, N * T)
    Cp = randn(N * T, K)
    u = randn(N * T)
    q = u - Cp * ζ
    return (; ζ, target_aggregate, q, u, Cp, C, S, obs_index)
end

# Independent O(N²) construction from the estimating equation. This deliberately
# does not call the production pair catalog or period-score helper.
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

@testset "masked empirical sandwich: orientation, scale, and pair scope" begin
    for excluded in (false, true)
        f = _vcov_scope_fixture(; excluded)
        _, Σ = solve_vcov(f.u, f.S, f.C, f.Cp, f.obs_index;
            precision=f.precision, Mweights=f.Mweights)
        ref = _independent_vcov(f.u, f.S, f.C, f.Cp, f.obs_index,
            f.precision, f.Mweights)
        @test Σ ≈ ref.Σ rtol=1e-11 atol=1e-11

        invG = inv(ref.G)
        reversed = invG' * ref.B * invG / f.obs_index.T
        @test norm(Σ - reversed) / norm(Σ) > 1e-2

        wrong_G = ref.G .* (f.obs_index.T / (f.obs_index.T - 1))
        wrong_scale = inv(wrong_G) * ref.B * inv(wrong_G)' / f.obs_index.T
        @test norm(Σ - wrong_scale) / norm(Σ) > 1e-2

        # The empirical period-score meat retains covariance across admissible
        # pair products; summing pairwise variances alone is observably different.
        @test norm(ref.B - ref.Bdiag) / norm(ref.B) > 1e-2
    end

    allpairs = _vcov_scope_fixture()
    excluded = _vcov_scope_fixture(; excluded=true)
    _, Σ_all = solve_vcov(allpairs.u, allpairs.S, allpairs.C, allpairs.Cp,
        allpairs.obs_index; precision=allpairs.precision, Mweights=allpairs.Mweights)
    _, Σ_excluded = solve_vcov(excluded.u, excluded.S, excluded.C, excluded.Cp,
        excluded.obs_index; precision=excluded.precision, Mweights=excluded.Mweights)
    @test norm(Σ_all - Σ_excluded) / norm(Σ_all) > 1e-2
end

@testset "period weighting and moment-score identity" begin
    f = _vcov_scope_fixture(; excluded=true)
    ζ = [0.7, 1.1]
    q = f.u - f.Cp * ζ
    scores, G = masked_period_scores(f.u, f.S, f.C, f.Cp, f.obs_index,
        f.precision, f.Mweights; bread=true)
    kernel_scores = moment_conditions(ζ, q, f.Cp, f.C, f.S, f.obs_index,
        true, Val(:iv); precision=f.precision, Mweights=f.Mweights)
    @test scores ≈ kernel_scores rtol=1e-12 atol=1e-12

    Gfd = ForwardDiff.jacobian(z -> mean_moment_conditions(z, q, f.Cp, f.C,
        f.S, f.obs_index, true, Val(:iv); precision=f.precision,
        Mweights=f.Mweights), ζ)
    @test G ≈ Gfd rtol=1e-10 atol=1e-10

    uniform = fill(1 / f.obs_index.T, f.obs_index.T)
    _, Σ_equal = solve_vcov(f.u, f.S, f.C, f.Cp, f.obs_index;
        precision=f.precision)
    _, Σ_uniform = solve_vcov(f.u, f.S, f.C, f.Cp, f.obs_index;
        precision=f.precision, Mweights=uniform)
    _, Σ_nonuniform = solve_vcov(f.u, f.S, f.C, f.Cp, f.obs_index;
        precision=f.precision, Mweights=f.Mweights)
    @test Σ_equal ≈ Σ_uniform rtol=1e-12 atol=1e-12
    @test norm(Σ_uniform - Σ_nonuniform) / norm(Σ_uniform) > 1e-3
end

@testset "excluded CUE uses full masked Jacobian across coverage regimes" begin
    f = _vcov_scope_fixture(; excluded=true)
    ζ = [0.9, 1.2]
    q = f.u - f.Cp * ζ
    σu² = calculate_entity_variance(f.u, f.obs_index)
    for complete_coverage in (false, true)
        cue_map = z -> mean_moment_conditions(z, q, f.Cp, f.C, f.S,
            f.obs_index, complete_coverage, Val(:iv))
        Gfull = ForwardDiff.jacobian(cue_map, ζ)
        cue_scores = moment_conditions(ζ, q, f.Cp, f.C, f.S,
            f.obs_index, complete_coverage, Val(:iv))
        cue_scores .-= mean(cue_scores; dims=2)
        B = cue_scores * cue_scores' / f.obs_index.T
        invG = inv(Gfull)
        Σref = invG * B * invG' / f.obs_index.T

        _, Σ = solve_vcov(f.u, f.S, f.C, f.Cp, f.obs_index;
            ζ, complete_coverage)
        @test Σ ≈ Σref rtol=2e-6 atol=1e-8

        Mw = complete_coverage ? period_mweights(ζ, f.C, f.S, f.obs_index) : nothing
        _, Gfixed = masked_period_scores(f.u, f.S, f.C, f.Cp, f.obs_index,
            1 ./ σu², Mw; bread=true)
        @test norm(Gfull - Gfixed) / norm(Gfull) > 1e-3
    end
    _, Σoptimal = solve_optimal_vcov(ζ, f.u, f.S, f.C, f.obs_index)
    @test all(isfinite, Σoptimal)
end

@testset "complete-coverage CUE derivative stays on positive branch" begin
    f = _near_boundary_cue_fixture()
    aggregate = [dot(f.S[f.obs_index.start_indices[t]:f.obs_index.end_indices[t]],
        f.C[f.obs_index.start_indices[t]:f.obs_index.end_indices[t], :] * f.ζ)
        for t in 1:f.obs_index.T]
    @test aggregate ≈ f.target_aggregate atol=1e-12
    cue_map = z -> mean_moment_conditions(z, f.q, f.Cp, f.C, f.S,
        f.obs_index, true, Val(:iv))
    Gref = ForwardDiff.jacobian(cue_map, f.ζ)
    Gunsafe = _central_difference_jacobian(cue_map, f.ζ)
    Gsafe = _positive_domain_jacobian(cue_map, f.ζ, f.C, f.S, f.obs_index)
    @test norm(Gunsafe - Gref) / norm(Gref) > 0.1
    @test norm(Gsafe - Gref) / norm(Gref) < 1e-5

    cue_scores = moment_conditions(f.ζ, f.q, f.Cp, f.C, f.S,
        f.obs_index, true, Val(:iv))
    cue_scores .-= mean(cue_scores; dims=2)
    B = cue_scores * cue_scores' / f.obs_index.T
    Σref = inv(Gref) * B * inv(Gref)' / f.obs_index.T
    _, Σ = solve_vcov(f.u, f.S, f.C, f.Cp, f.obs_index;
        ζ=f.ζ, complete_coverage=true)
    @test Σ ≈ Σref rtol=2e-5 atol=1e-8
end

@testset "specialized estimators: cue-pinned vcov, mode/mask rejections" begin
    df = _load_simdata1()
    formula = @formula(q + endog(p) ~ 0 + fe(id) & (η1 + η2))
    scalar = giv(df, formula, :id, :t, :absS; guess=Dict("Aggregate" => 2.0),
        quiet=true, algorithm=:scalar_search, complete_coverage=true)
    debiased = giv(df, formula, :id, :t, :absS; guess=[1.0], quiet=true,
        algorithm=:debiased_ols, complete_coverage=true, precision_weights=:cue)
    @test coef(debiased) ≈ coef(scalar) atol=1e-8
    @test vcov(debiased) ≈ vcov(scalar) rtol=1e-8 atol=1e-12
    @test endog_coef(scalar)[1] * 2 ≈ 2.5341730 atol=1e-4
    @test stderror(scalar)[1] * 2 ≈ 0.2407 atol=1e-4

    # Decision 14: both full-market estimators report the information formula.
    @test scalar.vcov_method == :optimal
    @test debiased.vcov_method == :optimal
    _, deb_mats = build_error_function(df, formula, :id, :t, :absS;
        algorithm=:debiased_ols, complete_coverage=true, precision_weights=:cue)
    deb_u = deb_mats.uq + deb_mats.uCp * endog_coef(debiased)
    _, deb_ref = solve_optimal_vcov(endog_coef(debiased), deb_u, deb_mats.S,
        deb_mats.C, deb_mats.obs_index)
    @test vcov(debiased) == deb_ref

    # Decision 14: fixed-weight modes are rejected for :debiased_ols (including
    # the resolved :twostep default), and it has no pairwise sandwich.
    for pw in (:raw_onestep, :twostep, ones(5))
        @test_throws ArgumentError giv(df, formula, :id, :t, :absS; guess=[1.0],
            quiet=true, algorithm=:debiased_ols, complete_coverage=true,
            precision_weights=pw)
    end
    @test_throws ArgumentError giv(df, formula, :id, :t, :absS; guess=[1.0],
        quiet=true, algorithm=:debiased_ols, complete_coverage=true)
    @test_throws ArgumentError giv(df, formula, :id, :t, :absS; guess=[1.0],
        quiet=true, algorithm=:debiased_ols, complete_coverage=true,
        precision_weights=:cue, vcov=:sandwich)

    exclusions = Dict(1 => [2])
    @test_throws ArgumentError giv(df, formula, :id, :t, :absS;
        guess=[1.0], quiet=true, algorithm=:debiased_ols,
        complete_coverage=true, precision_weights=:cue, exclude_pairs=exclusions)
    @test_throws ArgumentError giv(df, formula, :id, :t, :absS;
        guess=Dict("Aggregate" => 2.0), quiet=true, algorithm=:scalar_search,
        complete_coverage=true, exclude_pairs=exclusions)
end

@testset "vcov route matrix" begin
    df = _load_simdata1()
    expected_coef = [1.596359996232405, 1.6570008433581191, 1.296430273277033,
        3.3349700649766336, 0.5844260250940619]
    expected_vcov = [
        3.176821524696577 -0.503741124188375 -0.2647564979059007 -0.1988746378021255 -0.22028747289049846
        -0.5037411241883748 0.23280346682930267 0.03543370928075544 0.026616404715024748 0.02948219339027695
        -0.2647564979059007 0.03543370928075546 0.15296661624768007 0.01398906255777693 0.01549526512287549
        -0.19887463780212553 0.026616404715024758 0.013989062557776932 0.1479449830106333 0.011639431943441995
        -0.22028747289049852 0.029482193390276964 0.015495265122875496 0.011639431943441993 0.03000245495060605
    ]
    clean = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true,
        algorithm=:iv, complete_coverage=true, precision_weights=:cue)
    @test endog_coef(clean) ≈ expected_coef rtol=1e-10 atol=1e-10
    @test vcov(clean) ≈ expected_vcov rtol=1e-10 atol=1e-10

    _, clean_mats = build_error_function(df, _FEQ, :id, :t, :absS;
        algorithm=:iv, complete_coverage=true, precision_weights=:cue)
    clean_u = clean_mats.uq + clean_mats.uCp * endog_coef(clean)
    _, clean_ref = solve_optimal_vcov(endog_coef(clean), clean_u,
        clean_mats.S, clean_mats.C, clean_mats.obs_index)
    @test vcov(clean) == clean_ref

    twostep = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true,
        algorithm=:iv, complete_coverage=true, precision_weights=:twostep)
    @test twostep.converged
    _, twostep_mats = build_error_function(df, _FEQ, :id, :t, :absS;
        algorithm=:iv, complete_coverage=true, precision_weights=:twostep)
    twostep_u = twostep_mats.uq + twostep_mats.uCp * endog_coef(twostep)
    _, twostep_ref = solve_optimal_vcov(endog_coef(twostep), twostep_u,
        twostep_mats.S, twostep_mats.C, twostep_mats.obs_index)
    @test vcov(twostep) == twostep_ref

    exclusions = Dict(1 => [2])
    excluded = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true,
        algorithm=:iv, complete_coverage=true, precision_weights=:cue,
        exclude_pairs=exclusions)
    _, excluded_mats = build_error_function(df, _FEQ, :id, :t, :absS;
        algorithm=:iv, complete_coverage=true, precision_weights=:cue,
        exclude_pairs=exclusions)
    excluded_u = excluded_mats.uq + excluded_mats.uCp * endog_coef(excluded)
    _, excluded_ref = solve_optimal_vcov(endog_coef(excluded), excluded_u,
        excluded_mats.S, excluded_mats.C, excluded_mats.obs_index)
    @test vcov(excluded) == excluded_ref

    excluded_twostep = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true,
        algorithm=:iv, complete_coverage=true, precision_weights=:twostep,
        exclude_pairs=exclusions)
    @test excluded_twostep.converged
    _, excluded_twostep_mats = build_error_function(df, _FEQ, :id, :t, :absS;
        algorithm=:iv, complete_coverage=true, precision_weights=:twostep,
        exclude_pairs=exclusions)
    excluded_twostep_u = excluded_twostep_mats.uq + excluded_twostep_mats.uCp *
        endog_coef(excluded_twostep)
    _, excluded_twostep_ref = solve_optimal_vcov(endog_coef(excluded_twostep),
        excluded_twostep_u, excluded_twostep_mats.S, excluded_twostep_mats.C,
        excluded_twostep_mats.obs_index)
    @test vcov(excluded_twostep) == excluded_twostep_ref

    excluded_precision = 1 ./ calculate_entity_variance(excluded_u, excluded_mats.obs_index)
    excluded_Mweights = period_mweights(endog_coef(excluded), excluded_mats.C,
        excluded_mats.S, excluded_mats.obs_index)
    excluded_scores, _ = masked_period_scores(excluded_u, excluded_mats.S,
        excluded_mats.C, excluded_mats.uCp, excluded_mats.obs_index,
        excluded_precision, excluded_Mweights)
    excluded_kernel = mean_moment_conditions(endog_coef(excluded), excluded_mats.uq,
        excluded_mats.uCp, excluded_mats.C, excluded_mats.S,
        excluded_mats.obs_index, true, Val(:iv))
    @test vec(mean(excluded_scores; dims=2)) ≈ excluded_kernel rtol=1e-12 atol=1e-12

    incomplete = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true,
        algorithm=:iv, complete_coverage=false, precision_weights=:cue)
    incomplete_u = clean_mats.uq + clean_mats.uCp * endog_coef(incomplete)
    _, incomplete_ref = solve_vcov(incomplete_u, clean_mats.S, clean_mats.C,
        clean_mats.uCp, clean_mats.obs_index; ζ=endog_coef(incomplete),
        complete_coverage=false)
    @test vcov(incomplete) == incomplete_ref

    fixed = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true,
        algorithm=:iv, complete_coverage=true, precision_weights=:raw_onestep)
    _, fixed_mats = build_error_function(df, _FEQ, :id, :t, :absS;
        algorithm=:iv, complete_coverage=true, precision_weights=:raw_onestep)
    fixed_u = fixed_mats.uq + fixed_mats.uCp * endog_coef(fixed)
    fixed_scores, _ = masked_period_scores(fixed_u, fixed_mats.S, fixed_mats.C,
        fixed_mats.uCp, fixed_mats.obs_index, fixed_mats.precision, nothing)
    fixed_kernel = mean_moment_conditions(endog_coef(fixed), fixed_mats.uq,
        fixed_mats.uCp, fixed_mats.C, fixed_mats.S, fixed_mats.obs_index, true,
        Val(:iv); precision=fixed_mats.precision)
    @test vec(mean(fixed_scores; dims=2)) ≈ fixed_kernel rtol=1e-12 atol=1e-12
    _, fixed_ref = solve_vcov(fixed_u, fixed_mats.S, fixed_mats.C,
        fixed_mats.uCp, fixed_mats.obs_index; precision=fixed_mats.precision)
    @test vcov(fixed) == fixed_ref
end

# ---------------------------------------------------------------------------
# `vcov` selector keyword, input validation, NaN-vcov return, recorded fields
# (giv-solver-stability/vcov-selector-hardening; umbrella decisions 13, 14, 16)
# ---------------------------------------------------------------------------

@testset "vcov selector: eligibility cells and :sandwich frozen-bundle equality" begin
    df = _load_simdata1()

    # frozen step-1 bundle used by the complete-coverage :twostep sandwich
    m1 = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true, algorithm=:iv,
        complete_coverage=true, precision_weights=:raw_onestep)
    _, mats = build_error_function(df, _FEQ, :id, :t, :absS; algorithm=:iv,
        complete_coverage=true, precision_weights=:raw_onestep)
    û₁ = mats.uq + mats.uCp * endog_coef(m1)
    w2 = 1 ./ calculate_entity_variance(û₁, mats.obs_index)
    Mw1 = period_mweights(endog_coef(m1), mats.C, mats.S, mats.obs_index)

    # --- complete-coverage :twostep: eligible for the information formula ---
    tw_auto = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true, algorithm=:iv,
        complete_coverage=true, precision_weights=:twostep)
    tw_opt = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true, algorithm=:iv,
        complete_coverage=true, precision_weights=:twostep, vcov=:optimal)
    tw_sand = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true, algorithm=:iv,
        complete_coverage=true, precision_weights=:twostep, vcov=:sandwich)
    @test tw_auto.vcov_method == :optimal
    @test tw_opt.vcov_method == :optimal
    @test vcov(tw_auto) == vcov(tw_opt)          # :auto ≡ :optimal when eligible
    @test tw_sand.vcov_method == :sandwich
    @test vcov(tw_auto) != vcov(tw_sand)          # different route ⇒ different SEs
    # :sandwich equals a hand-built solve_vcov on the frozen step-1 weight bundle
    û₂ = mats.uq + mats.uCp * endog_coef(tw_sand)
    _, Σ_frozen = solve_vcov(û₂, mats.S, mats.C, mats.uCp, mats.obs_index;
        precision=w2, Mweights=Mw1)
    @test vcov(tw_sand) == Σ_frozen

    # --- complete-coverage :cue: also eligible ---
    cue_auto = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true, algorithm=:iv,
        complete_coverage=true, precision_weights=:cue)
    cue_sand = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true, algorithm=:iv,
        complete_coverage=true, precision_weights=:cue, vcov=:sandwich)
    @test cue_auto.vcov_method == :optimal
    @test cue_sand.vcov_method == :sandwich
    cue_u = mats.uq + mats.uCp * endog_coef(cue_sand)
    _, Σ_cue_sand = solve_vcov(cue_u, mats.S, mats.C, mats.uCp, mats.obs_index;
        ζ=endog_coef(cue_sand), complete_coverage=true)
    @test vcov(cue_sand) == Σ_cue_sand

    # --- complete-coverage :raw_onestep and custom vector: sandwich-only ---
    for pw in (:raw_onestep, w2)
        auto = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true, algorithm=:iv,
            complete_coverage=true, precision_weights=pw)
        @test auto.vcov_method == :sandwich
        @test_throws ArgumentError giv(df, _FEQ, :id, :t, :absS; guess=ones(5),
            quiet=true, algorithm=:iv, complete_coverage=true,
            precision_weights=pw, vcov=:optimal)
    end

    # --- incomplete coverage: never information-formula eligible ---
    for pw in (:twostep, :cue)
        inc = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true, algorithm=:iv,
            complete_coverage=false, precision_weights=pw)
        @test inc.vcov_method == :sandwich
        @test_throws ArgumentError giv(df, _FEQ, :id, :t, :absS; guess=ones(5),
            quiet=true, algorithm=:iv, complete_coverage=false,
            precision_weights=pw, vcov=:optimal)
    end
end

@testset "vcov selector: recorded reproducibility fields" begin
    df = _load_simdata1()
    m1 = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true, algorithm=:iv,
        complete_coverage=true, precision_weights=:raw_onestep)
    _, mats = build_error_function(df, _FEQ, :id, :t, :absS; algorithm=:iv,
        complete_coverage=true, precision_weights=:raw_onestep)
    û₁ = mats.uq + mats.uCp * endog_coef(m1)
    w2 = 1 ./ calculate_entity_variance(û₁, mats.obs_index)
    Mw1 = period_mweights(endog_coef(m1), mats.C, mats.S, mats.obs_index)

    tw = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true, algorithm=:iv,
        complete_coverage=true, precision_weights=:twostep)
    @test tw.precision_weights === :twostep
    @test tw.entity_precision == w2              # frozen step-2 precisions
    @test tw.period_weights == Mw1               # frozen complete-coverage multipliers
    @test tw.vcov_method == :optimal

    raw = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true, algorithm=:iv,
        complete_coverage=true, precision_weights=:raw_onestep)
    @test raw.precision_weights === :raw_onestep
    @test raw.entity_precision == mats.precision
    @test isnothing(raw.period_weights)          # equal period weights
    @test raw.vcov_method == :sandwich

    cue = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true, algorithm=:iv,
        complete_coverage=true, precision_weights=:cue)
    @test cue.precision_weights === :cue
    @test isnothing(cue.entity_precision)        # not frozen
    @test isnothing(cue.period_weights)
    @test cue.vcov_method == :optimal

    custom = giv(df, _FEQ, :id, :t, :absS; guess=ones(5), quiet=true, algorithm=:iv,
        complete_coverage=true, precision_weights=w2)
    @test custom.precision_weights == w2
    @test custom.entity_precision == w2
    @test custom.vcov_method == :sandwich
end

@testset "vcov selector: input validation" begin
    df = _load_simdata1()
    _, mats = build_error_function(df, _FEQ, :id, :t, :absS; algorithm=:iv,
        complete_coverage=true, precision_weights=:raw_onestep)
    N = length(mats.precision)
    good = fill(2.0, N)

    bad_inf = copy(good); bad_inf[1] = Inf
    bad_nan = copy(good); bad_nan[2] = NaN
    bad_neg = copy(good); bad_neg[1] = -1.0
    bad_zero = copy(good); bad_zero[1] = 0.0
    for badvec in (bad_inf, bad_nan, bad_neg, bad_zero)
        @test_throws ArgumentError giv(df, _FEQ, :id, :t, :absS; guess=ones(5),
            quiet=true, algorithm=:iv, complete_coverage=true, precision_weights=badvec)
    end

    # unknown vcov selector
    @test_throws ArgumentError giv(df, _FEQ, :id, :t, :absS; guess=ones(5),
        quiet=true, algorithm=:iv, complete_coverage=true, vcov=:robust)

    # scalar_search rejects an explicit precision_weights; omission is fine
    sform = @formula(q + endog(p) ~ 0 + fe(id) & (η1 + η2))
    @test_throws ArgumentError giv(df, sform, :id, :t, :absS;
        guess=Dict("Aggregate" => 2.0), quiet=true, algorithm=:scalar_search,
        complete_coverage=true, precision_weights=:cue)
    s = giv(df, sform, :id, :t, :absS; guess=Dict("Aggregate" => 2.0), quiet=true,
        algorithm=:scalar_search, complete_coverage=true)
    @test isnothing(s.precision_weights) && isnothing(s.entity_precision)
end

# Complete-coverage CUE landing at a nonpositive/near-zero aggregate elasticity:
# both the sandwich derivative and the information formula require the positive
# domain. Random symmetric flows with a negative starting guess drive the CUE
# root off-domain (seed pinned, as elsewhere in this suite).
function _offdomain_cue_fixture()
    Random.seed!(5)
    N, T = 3, 25
    ids = repeat(1:N, T)
    ts = repeat(1:T; inner=N)
    S = fill(1.0 / N, N * T)
    p = repeat(randn(T); inner=N)
    q = randn(N * T)
    for t in 1:T
        r = (t - 1) * N + 1:t * N
        q[r] .-= sum(S[r] .* q[r]) / sum(S[r])   # enforce exact market clearing
    end
    df = DataFrame(id=CategoricalArray(ids), t=ts, q=q, p=p, S=S, η=randn(N * T))
    return df, @formula(q + endog(p) ~ 0 + η)
end

@testset "vcov selector: off-domain CUE returns NaN vcov without crashing" begin
    df, f = _offdomain_cue_fixture()
    # giv must NOT let the DomainError from the CUE sandwich derivative escape.
    m = giv(df, f, :id, :t, :S; guess=[-3.0], quiet=true, algorithm=:iv,
        complete_coverage=true, precision_weights=:cue, iterations=200)
    _, mats = build_error_function(df, f, :id, :t, :S; algorithm=:iv,
        complete_coverage=true, precision_weights=:cue)
    @test !aggregate_elasticity_in_domain(endog_coef(m), mats.C, mats.S, mats.obs_index)
    @test !m.converged
    @test all(isnan, vcov(m))
    @test m.vcov_method == :none

    # Direct solve_vcov at the same off-domain root keeps its throwing behavior.
    û = mats.uq + mats.uCp * endog_coef(m)
    @test_throws DomainError solve_vcov(û, mats.S, mats.C, mats.uCp, mats.obs_index;
        ζ=endog_coef(m), complete_coverage=true)
end
