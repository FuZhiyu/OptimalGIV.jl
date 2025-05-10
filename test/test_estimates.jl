using Test, GIV
using DataFrames, CSV
# Random.seed!(6)
# simmodel = GIV.SimModel(T = 60, N = 4, varᵤshare = 0.8, usupplyshare = 0.2, h = 0.3, σᵤcurv = 0.2, ζs = 0.5, NC = 2, M = 0.5)
# df = DataFrame(simmodel.data)
# CSV.write("$(@__DIR__)/../examples/simdata1.csv", df)
df = CSV.read("$(@__DIR__)/../examples/simdata1.csv", DataFrame)

#============== assuming homogeneous elasticity ==============#
givmodel = giv(
    df,
    @formula(q + endog(p) ~ id & (η1 + η2)),
    :id,
    :t,
    :absS;
    guess = Dict("Aggregate" => 2.0),
    algorithm = :scalar_search,
)

f = @formula(q + endog(p) ~ id & (η1 + η2))
response_term, slope_terms, endog_term, exog_terms = GIV.parse_endog(f)

# est = estimate_model(simmodel.data, coefmapping = ones(Bool, 5, 1), ζSguess = 2.0,)
# 1/est.M[1]
@test givmodel.coef[1] * 2 ≈ 2.5341730 atol = 1e-4

# delta_method(θ̂->[abs.(simmodel.data.S[:, 1])' * θ̂], est.ζ, est.Σζ)
@test sqrt.(givmodel.vcov)[1] * 2 ≈ 0.2407 atol = 1e-4

factor_coef = [0.2419 1.1729; 0.1842 0.3722; -1.3213 -0.6487; 0.6288 -0.4422; 0.7269 1.6341]
@test givmodel.factor_coef ≈ vec(factor_coef) atol = 1e-4

# algorithm uu
givmodel_uu = giv(
    df,
    @formula(q + endog(p) ~ id & (η1 + η2)),
    :id,
    :t,
    :absS;
    guess = Dict("Constant" => 1.0),
    algorithm=:iv,
    savedf = true,
)
@test givmodel_uu.coef ≈ givmodel.coef atol = 1e-6

# algorithm up
givmodel_up = giv(
    df,
    @formula(q + endog(p) ~ id & (η1 + η2)),
    :id,
    :t,
    :absS;
    guess = Dict("Constant" => 1.0),
    algorithm=:debiased_ols,
)
@test givmodel_up.coef ≈ givmodel.coef atol = 1e-6

#============== assuming heterogeneous elasticity ==============#
givmodel = giv(
    df,
    @formula(q + id & endog(p) ~ id & (η1 + η2)),
    :id,
    :t,
    :absS;
    guess = Dict("Aggregate" => 2.5),
    algorithm = :scalar_search,
    savedf = true,
)
# est = estimate_model(simmodel.data,  ζSguess = 2.5,)
# println(round.(est.ζ, digits = 5))
@test givmodel.coef ≈ [1.59636, 1.657, 1.29643, 3.33497, 0.58443] atol = 1e-4
# println(round.(sqrt.(diag(est.Σζ)), digits = 4))
givse = sqrt.(GIV.diag(givmodel.vcov))
@test givse ≈ [1.7824, 0.4825, 0.3911, 0.3846, 0.1732] atol = 1e-4

# println(round.(vec(est.m), digits = 4))
factor_coef = [0.3406, 0.301, -1.3125, 1.2485, 0.5224, 1.4531, 0.7041, -0.6237, 1.3181, 1.053]
@test givmodel.factor_coef ≈ factor_coef atol = 1e-4

givmodel_uu = giv(
    df,
    @formula(q + id & endog(p) ~ id & (η1 + η2)),
    :id,
    :t,
    :absS;
    guess = Dict("id" => ones(5)),
    algorithm=:iv,
)
@test givmodel_uu.coef ≈ givmodel.coef atol = 1e-6

givmodel_up = giv(
    df,
    @formula(q + id & endog(p) ~ id & (η1 + η2)),
    :id,
    :t,
    :absS;
    guess = Dict("id" => ones(5)),
    algorithm=:debiased_ols,
)
@test givmodel_up.coef ≈ givmodel.coef atol = 1e-6

#============== exclude certain sectors ==============#
subdf = subset(df, :id => (x -> x .> 1))
givmodel = giv(
    subdf,
    @formula(q + id & endog(p) ~ id & (η1 + η2)),
    :id,
    :t,
    :absS;
    guess = Dict("Aggregate" => 2.0),
    algorithm = :scalar_search,
)

# est = estimate_model(simmodel.data, ζSguess = 2.0, exclude_categories = [1])
# println(round.(est.ζ[2:5], digits = 4))
@test givmodel.coef ≈ [1.9772, 1.4518, 3.4499, 0.7464] atol = 1e-4

givmodel_up = giv(
    subdf,
    @formula(q + id & endog(p) ~ id & (η1 + η2)),
    :id,
    :t,
    :absS;
    guess = Dict("id" => ones(4)),
    algorithm=:debiased_ols,
)
@test givmodel_up.coef ≈ givmodel.coef atol = 1e-6

givmodel_uu = giv(
    subdf,
    @formula(q + id & endog(p) ~ id & (η1 + η2)),
    :id,
    :t,
    :absS;
    guess=Dict("id" => 1.5 * ones(4)),
    algorithm=:iv_legacy,
)
@test givmodel_uu.coef ≈ [1.0442, 0.9967, 4.2707, 0.7597] atol = 1e-4
