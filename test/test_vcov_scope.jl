using Test, OptimalGIV, Random, LinearAlgebra, ForwardDiff, Statistics
using OptimalGIV: calculate_entity_variance, create_observation_index,
    masked_period_scores, mean_moment_conditions, moment_conditions,
    period_mweights, solve_optimal_vcov, solve_specialized_vcov, solve_vcov,
    _central_difference_jacobian, _positive_domain_jacobian
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

@testset "specialized estimators preserve no-exclusion vcov and reject masks" begin
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

    fixed = giv(df, formula, :id, :t, :absS; guess=[1.0], quiet=true,
        algorithm=:debiased_ols, complete_coverage=true,
        precision_weights=:raw_onestep)
    _, fixed_mats = build_error_function(df, formula, :id, :t, :absS;
        algorithm=:debiased_ols, complete_coverage=true,
        precision_weights=:raw_onestep)
    fixed_u = fixed_mats.uq + fixed_mats.uCp * endog_coef(fixed)
    _, fixed_ref = solve_specialized_vcov(fixed_u, fixed_mats.S, fixed_mats.C,
        fixed_mats.uCp, fixed_mats.obs_index; precision=fixed_mats.precision)
    @test vcov(fixed) == fixed_ref
    # Frozen regression oracle from pre-change commit 5f94698's fixed-weight
    # `solve_vcov` route; the copied compatibility helper above independently
    # reproduces that legacy calculation from the fitted residuals.
    @test endog_coef(fixed)[1] ≈ 1.0627621672644294 rtol=1e-10
    @test vcov(fixed)[1, 1] ≈ 0.016200219034447814 rtol=1e-10

    exclusions = Dict(1 => [2])
    @test_throws ArgumentError giv(df, formula, :id, :t, :absS;
        guess=[1.0], quiet=true, algorithm=:debiased_ols,
        complete_coverage=true, precision_weights=:cue, exclude_pairs=exclusions)
    @test_throws ArgumentError giv(df, formula, :id, :t, :absS;
        guess=Dict("Aggregate" => 2.0), quiet=true, algorithm=:scalar_search,
        complete_coverage=true, exclude_pairs=exclusions)
end

@testset "vcov route matrix and clean CUE pin" begin
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

    pin_df, pin_formula, ζtrue = _timevarying_complete_fixture()
    _, _, endog_names, _, _ = OptimalGIV.get_coefnames(pin_df, pin_formula)
    pinned = giv(pin_df, pin_formula, :id, :t, :S; guess=ζtrue, quiet=true,
        algorithm=:iv, complete_coverage=true, precision_weights=:cue,
        pin_zero=[endog_names[1]])
    @test pinned.converged
    @test endog_coef(pinned)[1] == 0.0
    @test all(iszero, vcov(pinned)[1, :]) && all(iszero, vcov(pinned)[:, 1])
    _, pin_mats = build_error_function(pin_df, pin_formula, :id, :t, :S;
        algorithm=:iv, complete_coverage=true, precision_weights=:cue,
        pin_zero=[endog_names[1]])
    ζpin_free = endog_coef(pinned)[2:end]
    pin_u = pin_mats.uq + pin_mats.uCp * ζpin_free
    _, pin_ref = solve_optimal_vcov(ζpin_free, pin_u, pin_mats.S,
        pin_mats.C, pin_mats.obs_index)
    @test vcov(pinned)[2:end, 2:end] == pin_ref

    pinned_twostep = giv(pin_df, pin_formula, :id, :t, :S; guess=ζtrue,
        quiet=true, algorithm=:iv, complete_coverage=true,
        precision_weights=:twostep, pin_zero=[endog_names[1]])
    @test pinned_twostep.converged
    @test endog_coef(pinned_twostep)[1] == 0.0
    @test all(iszero, vcov(pinned_twostep)[1, :]) &&
          all(iszero, vcov(pinned_twostep)[:, 1])
    ζpin_twostep_free = endog_coef(pinned_twostep)[2:end]
    pin_twostep_u = pin_mats.uq + pin_mats.uCp * ζpin_twostep_free
    _, pin_twostep_ref = solve_optimal_vcov(ζpin_twostep_free,
        pin_twostep_u, pin_mats.S, pin_mats.C, pin_mats.obs_index)
    @test vcov(pinned_twostep)[2:end, 2:end] == pin_twostep_ref
end
