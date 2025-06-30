using Test, OptimalGIV, Random, LinearAlgebra
using OptimalGIV: moment_conditions, ObservationIndex, create_observation_index
using DataFrames, CSV

@testset "moment_conditions iv vs iv_legacy" begin
    # Set random seed for reproducibility
    Random.seed!(12345)

    # Create test data
    N, T = 20, 5
    Nmom = 3

    # Create test DataFrame for proper observation index creation
    df = DataFrame(
        id=repeat(1:N, outer=T),
        t=repeat(1:T, inner=N),
        dummy=ones(N * T)  # Dummy column for DataFrame completeness
    )

    # Generate random data
    q = randn(N * T)
    Cp = randn(N * T, Nmom)
    C = randn(N * T, Nmom)
    S = randn(N * T)


    # Test case 1: No exclusions
    @testset "No exclusions" begin
        # Use existing function to create observation index
        obs_index = create_observation_index(df, :id, :t)

        for _ in 1:3
            ζ = randn(Nmom)

            # Compute moment conditions using both methods
            err_legacy = moment_conditions(ζ, q, Cp, C, S, obs_index, true, Val(:iv_twopass))
            err_new = moment_conditions(ζ, q, Cp, C, S, obs_index, true, Val(:iv))

            # Test that the results are identical within numerical precision
            @test size(err_legacy) == size(err_new)
            @test all(isapprox.(err_legacy, err_new, rtol=1e-10))
        end
    end

    # Test case 2: With exclusions
    @testset "With exclusions" begin
        # Create exclusion pairs dictionary
        exclude_pairs = Dict{Int,Vector{Int}}()
        for i in 1:N
            for j in i+1:N
                # Exclude about 20% of pairs randomly
                if rand() < 0.2
                    if !haskey(exclude_pairs, i)
                        exclude_pairs[i] = Int[]
                    end
                    push!(exclude_pairs[i], j)
                end
            end
        end
        # Use existing function to create observation index with exclusions
        obs_index = create_observation_index(df, :id, :t, exclude_pairs)

        for _ in 1:3
            ζ = randn(Nmom)

            # Compute moment conditions using both methods
            err_legacy = moment_conditions(ζ, q, Cp, C, S, obs_index, true, Val(:iv_twopass))
            err_new = moment_conditions(ζ, q, Cp, C, S, obs_index, true, Val(:iv))

            # Test that the results are identical within numerical precision
            @test size(err_legacy) == size(err_new)
            @test all(isapprox.(err_legacy, err_new, rtol=1e-10))
        end
    end
end


@testset "PC extraction equivalence" begin
    # Set random seed for reproducibility
    Random.seed!(12345)

    # Create test data
    N, T = 20, 5
    Nmom = 3
    n_pcs = 2  # Number of principal components to extract

    # Create test DataFrame for proper observation index creation
    df = DataFrame(
        id=repeat(1:N, outer=T),
        t=repeat(1:T, inner=N),
        dummy=ones(N * T)  # Dummy column for DataFrame completeness
    )

    # Generate random data
    q = randn(N * T)
    Cp = randn(N * T, Nmom)
    C = randn(N * T, Nmom)
    S = randn(N * T)

    # Test case 1: No exclusions with PC extraction
    @testset "No exclusions with PC extraction" begin
        # Use existing function to create observation index
        obs_index = create_observation_index(df, :id, :t)

        for _ in 1:3
            ζ = randn(Nmom)

            # Compute moment conditions using both methods with PC extraction
            err_legacy = moment_conditions(ζ, q, Cp, C, S, obs_index, true, Val(:iv_twopass), n_pcs)
            err_new = moment_conditions(ζ, q, Cp, C, S, obs_index, true, Val(:iv), n_pcs)

            # Test that the results are identical within numerical precision
            @test size(err_legacy) == size(err_new)
            @test all(isapprox.(err_legacy, err_new, rtol=1e-10))
        end
    end

    # Test case 2: With exclusions and PC extraction
    @testset "With exclusions and PC extraction" begin
        # Create exclusion pairs dictionary
        exclude_pairs = Dict{Int,Vector{Int}}()
        for i in 1:N
            for j in i+1:N
                # Exclude about 20% of pairs randomly
                if rand() < 0.2
                    if !haskey(exclude_pairs, i)
                        exclude_pairs[i] = Int[]
                    end
                    push!(exclude_pairs[i], j)
                end
            end
        end
        # Use existing function to create observation index with exclusions
        obs_index = create_observation_index(df, :id, :t, exclude_pairs)

        for _ in 1:3
            ζ = randn(Nmom)

            # Compute moment conditions using both methods with PC extraction
            err_legacy = moment_conditions(ζ, q, Cp, C, S, obs_index, true, Val(:iv_twopass), n_pcs)
            err_new = moment_conditions(ζ, q, Cp, C, S, obs_index, true, Val(:iv), n_pcs)

            # Test that the results are identical within numerical precision
            @test size(err_legacy) == size(err_new)
            @test all(isapprox.(err_legacy, err_new, rtol=1e-10))
        end
    end
end


@testset "standard error equivalence" begin
    using CategoricalArrays
    df = CSV.read("$(@__DIR__)/../examples/simdata1.csv", DataFrame)
    df.id = CategoricalArray(df.id)
    givmodel = giv(
        df,
        @formula(q + id & endog(p) ~ fe(id) & (η1 + η2) + 0), # numerical equivalence only holds when the endogenous variables are not included in the instrument
        :id,
        :t,
        :absS;
        guess=ones(5),
        algorithm=:iv_twopass,
    )

    givmodel2 = giv(
        df,
        @formula(q + id & endog(p) ~ fe(id) & (η1 + η2) + 0),
        :id,
        :t,
        :absS;
        guess=ones(5),
        algorithm=:iv_twopass,
        complete_coverage=false, # use the nonoptimal vcov algorithm
    )

    @test maximum(abs, coef(givmodel2) - coef(givmodel)) < 1e-6
    @test maximum(abs, vcov(givmodel2) - vcov(givmodel)) < 1e-6
end
