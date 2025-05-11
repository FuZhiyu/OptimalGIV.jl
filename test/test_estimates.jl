using Test, GIV
using DataFrames, CSV, CategoricalArrays
df = CSV.read("$(@__DIR__)/../examples/simdata1.csv", DataFrame)
df.id = CategoricalArray(df.id)
#============== assuming homogeneous elasticity ==============#
@testset "homogeneous elasticity" begin
    f = @formula(q + endog(p) ~ 0 + fe(id) & (η1 + η2))
    givmodel = giv(
        df,
        f,
        :id,
        :t,
        :absS;
        guess=Dict("Aggregate" => 2.0),
        save=:all,
        algorithm=:scalar_search,
    )

    @test givmodel.coef[1] * 2 ≈ 2.5341730 atol = 1e-4
    @test sqrt.(givmodel.vcov)[1] * 2 ≈ 0.2407 atol = 1e-4
    @test sqrt.(givmodel.vcov)[1] * 2 ≈ 0.2407 atol = 1e-4
    factor_coef = [0.2419 1.1729; 0.1842 0.3722; -1.3213 -0.6487; 0.6288 -0.4422; 0.7269 1.6341]
    @test maximum(abs.(Matrix(givmodel.coefdf[:, 3:4]) - factor_coef)) < 1e-4

    # use fe(id) to absorb one control
    givmodel_partialabsorbed = giv(
        df,
        @formula(q + endog(p) ~ 0 + id & η1 + fe(id) & η2),
        :id,
        :t,
        :absS;
        guess=Dict("Aggregate" => 2.0),
        algorithm=:scalar_search,
    )

    factor_coef = [0.2419 1.1729; 0.1842 0.3722; -1.3213 -0.6487; 0.6288 -0.4422; 0.7269 1.6341]
    @test givmodel_partialabsorbed.coef[2:end] ≈ factor_coef[:, 1] atol = 1e-4

    # algorithm uu
    givmodel_uu = giv(
        df,
        f,
        :id,
        :t,
        :absS;
        guess=[1.0],
        algorithm=:iv,
        # savedf=true,
    )
    @test givmodel_uu.coef ≈ givmodel.coef atol = 1e-6
    # algorithm up
    givmodel_up = giv(
        df,
        f,
        :id,
        :t,
        :absS;
        guess=[1.0],
        algorithm=:debiased_ols,
    )
    @test givmodel_up.coef ≈ givmodel.coef atol = 1e-6
end

#============== assuming heterogeneous elasticity ==============#
@testset "heterogeneous elasticity" begin
    f_het = @formula(q + id & endog(p) ~ 0 + id & (η1 + η2))
    givmodel = giv(
        df,
        f_het,
        :id,
        :t,
        :absS;
        guess=Dict("Aggregate" => 2.5),
        algorithm=:scalar_search,
    )
    # est = estimate_model(simmodel.data,  ζSguess = 2.5,)
    # println(round.(est.ζ, digits = 5))
    @test givmodel.coef[1:5] ≈ [1.59636, 1.657, 1.29643, 3.33497, 0.58443] atol = 1e-4
    # println(round.(sqrt.(diag(est.Σζ)), digits = 4))
    givse = sqrt.(GIV.diag(givmodel.vcov)[1:5])
    @test givse ≈ [1.7824, 0.4825, 0.3911, 0.3846, 0.1732] atol = 1e-4

    # println(round.(vec(est.m), digits = 4))
    factor_coef = [0.3406, 0.301, -1.3125, 1.2485, 0.5224, 1.4531, 0.7041, -0.6237, 1.3181, 1.053]
    @test givmodel.coef[6:end] ≈ factor_coef atol = 1e-4

    givmodel_uu = giv(
        df,
        f_het,
        :id,
        :t,
        :absS;
        guess=ones(5),
        algorithm=:iv,
    )
    @test givmodel_uu.coef ≈ givmodel.coef atol = 1e-6
    givmodel_up = giv(
        df,
        f_het,
        :id,
        :t,
        :absS;
        guess=ones(5),
        algorithm=:debiased_ols,
    )
    @test givmodel_up.coef ≈ givmodel.coef atol = 1e-6
end
# #============== exclude certain sectors ==============#
# subdf = subset(df, :id => (x -> (x) .> 1))
# givmodel = giv(
#     subdf,
#     @formula(q + id & endog(p) ~ id & (η1 + η2)),
#     :id,
#     :t,
#     :absS;
#     guess = Dict("Aggregate" => 2.0),
#     algorithm = :scalar_search,
# )

# # est = estimate_model(simmodel.data, ζSguess = 2.0, exclude_categories = [1])
# # println(round.(est.ζ[2:5], digits = 4))
# @test givmodel.coef ≈ [1.9772, 1.4518, 3.4499, 0.7464] atol = 1e-4

# givmodel_up = giv(
#     subdf,
#     @formula(q + id & endog(p) ~ id & (η1 + η2)),
#     :id,
#     :t,
#     :absS;
#     guess = Dict("id" => ones(4)),
#     algorithm=:debiased_ols,
# )
# @test givmodel_up.coef ≈ givmodel.coef atol = 1e-6

# givmodel_uu = giv(
#     subdf,
#     @formula(q + id & endog(p) ~ id & (η1 + η2)),
#     :id,
#     :t,
#     :absS;
#     guess=Dict("id" => 1.5 * ones(4)),
#     algorithm=:iv_legacy,
# )
# @test givmodel_uu.coef ≈ [1.0442, 0.9967, 4.2707, 0.7597] atol = 1e-4
