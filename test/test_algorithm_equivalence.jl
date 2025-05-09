using Test, GIV, Random, LinearAlgebra
using GIV: moment_conditions, ObservationIndex
using DataFrames, CSV

@testset "moment_conditions iv vs iv_legacy" begin
    # Set random seed for reproducibility
    Random.seed!(12345)

    # Create test data
    N, T = 20, 5
    Nmom = 3

    # Generate random data
    q = randn(N * T)
    Cp = randn(N * T, Nmom)
    C = randn(N * T, Nmom)
    S = randn(N * T)

    # Create observation index structure
    ids = repeat(1:N, outer=T)
    start_indices = [(t - 1) * N + 1 for t in 1:T]
    end_indices = [t * N for t in 1:T]

    # Create entity_obs_indices matrix for mapping entities to observation indices
    entity_obs_indices = zeros(Int, N, T)
    for t in 1:T
        for i in 1:N
            idx = (t - 1) * N + i
            entity_obs_indices[i, t] = idx
        end
    end

    obs_index = ObservationIndex(start_indices, end_indices, ids, entity_obs_indices, N, T)

    # Test case 1: No exclusions
    @testset "No exclusions" begin
        exclmat = BitArray(zeros(N, N))

        for _ in 1:3
            ζ = randn(Nmom)

            # Compute moment conditions using both methods
            err_legacy = moment_conditions(ζ, q, Cp, C, S, exclmat, obs_index, true, Val(:iv_legacy))
            err_new = moment_conditions(ζ, q, Cp, C, S, exclmat, obs_index, true, Val(:iv))

            # Test that the results are identical within numerical precision
            @test size(err_legacy) == size(err_new)
            @test all(isapprox.(err_legacy, err_new, rtol=1e-10))
        end
    end

    # Test case 2: With exclusions
    @testset "With exclusions" begin
        # Create exclusion matrix (sparse to test actual exclusions)
        exclmat = falses(N, N)
        for i in 1:N
            for j in i+1:N
                # Exclude about 20% of pairs randomly
                if rand() < 0.2
                    exclmat[i, j] = exclmat[j, i] = true
                end
            end
        end

        for _ in 1:3
            ζ = randn(Nmom)

            # Compute moment conditions using both methods
            err_legacy = moment_conditions(ζ, q, Cp, C, S, exclmat, obs_index, true, Val(:iv_legacy))
            err_new = moment_conditions(ζ, q, Cp, C, S, exclmat, obs_index, true, Val(:iv))

            # Test that the results are identical within numerical precision
            @test size(err_legacy) == size(err_new)
            @test all(isapprox.(err_legacy, err_new, rtol=1e-10))
        end
    end
end


@testset "standard error equivalence" begin
    df = CSV.read("$(@__DIR__)/../examples/simdata1.csv", DataFrame)
    givmodel = giv(
        df,
        @formula(q + id & endog(p) ~ id & (η1 + η2)), # numerical equivalence only holds when the endogenous variables are not included in the instrument
        :id,
        :t,
        :absS;
        guess=ones(5),
        algorithm=:iv_legacy,
    )
    
    givmodel2 = giv(
        df,
        @formula(q + id & endog(p) ~ id & (η1 + η2)),
        :id,
        :t,
        :absS;
        guess=ones(5),
        algorithm=:iv_legacy,
        universe_observed = false, # use the nonoptimal vcov algorithm
    )
    @test maximum(abs,givmodel2.coef - givmodel.coef) < 1e-6
    @test maximum(abs,givmodel2.vcov - givmodel.vcov) < 1e-6
end
