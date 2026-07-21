using Test, OptimalGIV, Random, LinearAlgebra
using OptimalGIV: build_error_function, diagnose_nonfinite_moment, NonfiniteMomentError,
    calculate_entity_variance, moment_conditions
using DataFrames, CSV, CategoricalArrays

# ---------------------------------------------------------------------------
# Nonfinite-moment guard (giv-solver-stability/nonfinite-forensics)
#
# The pre-2000 long-sample Treasury split failed with NLsolve's opaque
# IsFiniteException ("equation(s) … non-finite: [1..9]") because three zero-size
# entities produced a 0/0 flow, whose NaN poisoned the shared OLS-FE
# residualization so every entity's residual — and every moment — went NaN.
# The guard replaces that opaque error with a NonfiniteMomentError that names the
# offending entity/moment channel. These tests cover the three channels the
# diagnostic isolates: nonfinite residuals, degenerate precision, zero weight.
# ---------------------------------------------------------------------------

const _NF_SIMDATA1 = joinpath(@__DIR__, "..", "examples", "simdata1.csv")
_nf_load() = (df = CSV.read(_NF_SIMDATA1, DataFrame); df.id = CategoricalArray(df.id); df)
const _NF_FEQ = @formula(q + id & endog(p) ~ fe(id) & (η1 + η2) + 0)

@testset "diagnose_nonfinite_moment: named channels" begin
    df = _nf_load()
    _, mats = build_error_function(df, _NF_FEQ, :id, :t, :absS; algorithm=:iv, complete_coverage=true)
    obs_index = mats.obs_index
    N, T = obs_index.N, obs_index.T
    Nm = size(mats.C, 2)

    # baseline finite intermediates at a sane guess
    ζ = zeros(Nm)
    u = mats.uq .+ mats.uCp * ζ
    prec = 1 ./ calculate_entity_variance(u, obs_index)
    weightsum = fill(1.0, Nm, T)

    # Channel 1: nonfinite residuals u for a specific entity
    u_bad = copy(u)
    victim = 3
    u_bad[findfirst(==(victim), obs_index.ids)] = NaN
    e1 = diagnose_nonfinite_moment(u_bad, prec, weightsum, obs_index)
    @test e1 isa NonfiniteMomentError
    @test occursin("residuals u nonfinite", e1.msg)
    @test occursin(string(victim), e1.msg)

    # Channel 2: degenerate (Inf) precision for one entity
    prec_bad = copy(prec)
    prec_bad[2] = Inf
    e2 = diagnose_nonfinite_moment(u, prec_bad, weightsum, obs_index)
    @test occursin("precision", e2.msg)

    # Channel 3: a moment with zero total weight
    ws_bad = copy(weightsum)
    ws_bad[1, :] .= 0.0
    e3 = diagnose_nonfinite_moment(u, prec, ws_bad, obs_index)
    @test occursin("per-moment total weight", e3.msg)

    # showerror renders the named prefix
    @test occursin("NonfiniteMomentError", sprint(showerror, e1))
end

@testset "guard: NaN response → named error at the INITIAL evaluation, not mid-solve" begin
    # A single NaN in the response poisons the shared OLS-FE residualization,
    # mimicking the 0/0-flow bug. The guard now lives at the initial moment
    # evaluation inside estimate_giv — exactly where NLsolve would raise its opaque
    # IsFiniteException — so giv() surfaces the named diagnosis while the raw moment
    # closure (build_error_function) stays unguarded (returns nonfinite, no throw).
    df = _nf_load()
    df.q[findfirst(==(levels(df.id)[4]), df.id)] = NaN
    errfunc, mats = build_error_function(df, _NF_FEQ, :id, :t, :absS; algorithm=:iv, complete_coverage=true)
    ζ = zeros(size(mats.C, 2))

    # raw moment closure is NOT guarded — it returns nonfinite rather than throwing,
    # preserving NLsolve's mid-solve recovery semantics
    @test all(!isfinite, errfunc(ζ))

    # full giv() path throws the named error at the initial evaluation
    err = try
        giv(df, _NF_FEQ, :id, :t, :absS; algorithm=:iv, guess=ζ, quiet=true, complete_coverage=true)
        nothing
    catch e
        e
    end
    @test err isa NonfiniteMomentError
    @test occursin("residuals u nonfinite", err.msg)

    # the guard lives in the shared estimate_giv entry, so :iv_twopass gets the same
    # named diagnosis (no opaque IsFiniteException asymmetry between the IV paths)
    @test_throws NonfiniteMomentError giv(df, _NF_FEQ, :id, :t, :absS;
        algorithm=:iv_twopass, guess=ζ, quiet=true, complete_coverage=true)
end

@testset "guard: degenerate (Inf) precision poisons the shared normalization" begin
    # A single Inf entity precision (near-zero residual variance ⇒ 1/σ² = Inf) makes
    # that entity's moment weightsum Inf, so the shared normalization
    # momweight ./= sum(momweight) sends every moment row nonfinite at the initial
    # guess. Feed the Inf via the fixed precision-weight mode so it is deterministic.
    df = _nf_load()
    _, mats = build_error_function(df, _NF_FEQ, :id, :t, :absS; algorithm=:iv, complete_coverage=true)
    N = mats.obs_index.N
    w = ones(N)
    w[3] = Inf
    ζ = zeros(size(mats.C, 2))
    err = try
        giv(df, _NF_FEQ, :id, :t, :absS; algorithm=:iv, guess=ζ, quiet=true, complete_coverage=true,
            precision_weights=w)
        nothing
    catch e
        e
    end
    @test err isa NonfiniteMomentError
    @test occursin("precision", err.msg)
    @test occursin("per-moment total weight", err.msg)
end
