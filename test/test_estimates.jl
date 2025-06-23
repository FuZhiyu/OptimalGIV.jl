using Test, OptimalGIV
using DataFrames, CSV, CategoricalArrays, LinearAlgebra
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
    givse = sqrt.(diag(givmodel.vcov)[1:5])
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

#============== test build_error_function ==============#
@testset "build_error_function validation" begin
    using OptimalGIV: build_error_function

    # Test 1: Homogeneous elasticity with scalar_search
    @testset "scalar_search algorithm" begin
        f = @formula(q + endog(p) ~ 0 + fe(id) & (η1 + η2))

        # First estimate the model
        givmodel = giv(
            df, f, :id, :t, :absS;
            guess=Dict("Aggregate" => 2.0),
            algorithm=:scalar_search,
            tol=1e-6
        )

        # Build the error function
        err_func, components = build_error_function(
            df, f, :id, :t, :absS;
            algorithm=:scalar_search
        )
        # The aggregate elasticity (ζS) should yield zero error
        # For scalar_search, the error function takes ζS directly
        givmodel.agg_coef
        err = err_func(givmodel.agg_coef)
        @test abs(err[2]) < 1e-6  # Should be very close to zero
    end

    # Test 2: Homogeneous elasticity with iv algorithm
    @testset "iv algorithm" begin
        f = @formula(q + endog(p) ~ 0 + fe(id) & (η1 + η2))

        # Estimate the model
        givmodel = giv(
            df, f, :id, :t, :absS;
            guess=[1.0],
            algorithm=:iv,
            tol=1e-6
        )

        # Build the error function
        err_func, components = build_error_function(
            df, f, :id, :t, :absS;
            algorithm=:iv
        )

        # Extract just the elasticity coefficients (not the factor loadings)
        ζ = givmodel.coef

        # The error function should return zero for the estimated coefficients
        err = err_func(ζ)
        @test norm(err, Inf) < 1e-6  # Should be very close to zero vector
    end

    # Test 3: Homogeneous elasticity with debiased_ols algorithm
    @testset "debiased_ols algorithm" begin
        f = @formula(q + endog(p) ~ 0 + fe(id) & (η1 + η2))

        # Estimate the model
        givmodel = giv(
            df, f, :id, :t, :absS;
            guess=[1.0],
            algorithm=:debiased_ols,
        )

        # Build the error function
        err_func, components = build_error_function(
            df, f, :id, :t, :absS;
            algorithm=:debiased_ols
        )

        # Extract just the elasticity coefficients
        ζ = givmodel.coef

        # The error function should return zero for the estimated coefficients
        err = err_func(ζ)
        @test norm(err) < 1e-6  # Should be very close to zero vector
    end

    # Test 4: Heterogeneous elasticity with scalar_search
    @testset "heterogeneous elasticity - scalar_search" begin
        f_het = @formula(q + id & endog(p) ~ 0 + id & (η1 + η2))

        givmodel = giv(
            df, f_het, :id, :t, :absS;
            guess=Dict("Aggregate" => 2.5),
            algorithm=:scalar_search,
        )

        err_func, components = build_error_function(
            df, f_het, :id, :t, :absS;
            algorithm=:scalar_search
        )

        # For scalar_search with heterogeneous elasticity, 
        # the aggregate elasticity should still yield zero error
        err = err_func(givmodel.agg_coef)
        @test abs(err[2]) < 1e-10
    end

    # Test 5: Heterogeneous elasticity with iv algorithm
    @testset "heterogeneous elasticity - iv" begin
        f_het = @formula(q + id & endog(p) ~ 0 + fe(id) & (η1 + η2))

        givmodel = giv(
            df, f_het, :id, :t, :absS;
            guess=ones(5),
            algorithm=:iv,
            tol=1e-8,
        )

        err_func, components = build_error_function(
            df, f_het, :id, :t, :absS;
            algorithm=:iv
        )

        # Extract elasticity coefficients (first 5 for 5 categories)
        ζ = givmodel.coef

        err = err_func(ζ)
        @test norm(err, Inf) < 1e-8
    end

    # Test 6: Heterogeneous elasticity with debiased_ols
    @testset "heterogeneous elasticity - debiased_ols" begin
        f_het = @formula(q + id & endog(p) ~ 0 + fe(id) & (η1 + η2))

        givmodel = giv(
            df, f_het, :id, :t, :absS;
            guess=ones(5),
            algorithm=:debiased_ols,
            tol=1e-8,
        )

        err_func, components = build_error_function(
            df, f_het, :id, :t, :absS;
            algorithm=:debiased_ols
        )

        ζ = givmodel.coef
        err = err_func(ζ)
        @test norm(err, Inf) < 1e-8
    end

    # Test 7: Test that wrong coefficients give non-zero error
    @testset "wrong coefficients yield non-zero error" begin
        f = @formula(q + endog(p) ~ 0 + fe(id) & (η1 + η2))

        err_func, components = build_error_function(
            df, f, :id, :t, :absS;
            algorithm=:iv
        )

        # Test with wrong coefficient (should NOT be zero)
        wrong_ζ = [0.5]  # Arbitrary wrong value
        err = err_func(wrong_ζ)
        @test norm(err) > 1e-3  # Should be significantly different from zero
    end

end
