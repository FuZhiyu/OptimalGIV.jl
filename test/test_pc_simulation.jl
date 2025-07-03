using Test, OptimalGIV, Random, Statistics, DataFrames
using OptimalGIV: simulate_data, evaluation_metrics
using OptimalGIV.HeteroPCA: DeflatedHeteroPCA, StandardHeteroPCA, DiagonalDeletion
using StatsBase
using DataFramesMeta, FixedEffectModels

function estimate_standard_giv(df, k; pca_option=(; impute_method=:zero, demean=false, maxiter=100, algorithm=StandardHeteroPCA(), suppress_warnings=true, abstol=1e-8))
    df = @chain df begin
        groupby(:id)
        @combine(:precision = 1 ./ var(skipmissing(:q)))
        @transform!(:precision = :precision ./ sum(:precision))
        innerjoin(df, on=:id)
    end
    m = reg(df, @formula(q ~ fe(t)), weights=:precision, save=:residuals)
    obsindex = OptimalGIV.create_observation_index(df, :id, :t)
    _, _, _, u = OptimalGIV.extract_pcs_from_residuals(m.residuals, obsindex, k; pca_option...)
    df.u = u
    giv = @chain df begin
        groupby(:id)
        @transform!(:precision = 1 ./ var(skipmissing(:u)))
        @transform!(:precision = :precision ./ sum(:precision))
        groupby(:t)
        @combine(:z = sum(skipmissing(:S .* :u)), :qE = mean(:q, weights(:precision)), :p = first(:p))
    end
    reg(giv, @formula(qE ~ (p ~ z)))
end
@testset "PC Factor Recovery Simulation" begin
    Random.seed!(123)
    
    # Simple simulation parameters for homogeneous elasticities
    simparams = (
        N=20,         # Number of entities
        T=200,         # Number of time periods  
        K=2,          # Number of common factors
        M=0.5,        # Aggregate elasticity (1/M = 2.0 true elasticity)
        σζ=0.0,       # Homogeneous elasticity (no heterogeneity)
        σp=2,       # Price volatility
        σᵤcurv=0.1,   # Size-dependent shock curvature
        h=0.2,        # Excess HHI
        ushare=0.3,   # Share of idiosyncratic shocks
        missingperc=0.0  # No missing data
    )

    # Number of simulations
    Nsims = 50
    println("Running $Nsims simulations comparing factor control methods...")

    # Storage for results
    results_true_factors = Float64[]
    results_pc_extraction = Float64[]
    results_pc_standard = Float64[]
    results_pc_diagonal = Float64[]
    results_no_factors = Float64[]
    results_standard_giv = Float64[]

    convergence_true = Int[]
    convergence_pc = Int[]
    convergence_pc_standard = Int[]
    convergence_pc_diagonal = Int[]
    convergence_no = Int[]
    convergence_standard_giv = Int[]

    for sim in 1:Nsims
        if sim % 10 == 0
            println("  Simulation $sim/$Nsims")
        end

        # Generate data
        simulated_dfs = simulate_data(simparams, Nsims=1, seed=sim)
        df = simulated_dfs[1]
        df.id = string.(df.id)  # Convert to string for consistency
        sort!(df, [:t, :id])
        
        true_elasticity = 1.0 / simparams.M  # Should be 2.0
        guess = 1.0

        try
            # Method 1: True factors (baseline)
            model_true = giv(
                df,
                @formula(q + endog(p) ~ 0 + id & (η1 + η2)),
                :id, :t, :S;
                guess=[guess],
                quiet=true,
                complete_coverage=false,
                algorithm=:iv
            )

            push!(convergence_true, model_true.converged ? 1 : 0)
            if model_true.converged
                push!(results_true_factors, endog_coef(model_true)[1])
            else
                push!(results_true_factors, NaN)
            end

        catch e
            push!(convergence_true, 0)
            push!(results_true_factors, NaN)
        end

        try
            # Method 2: PC extraction
            model_pc = giv(
                df,
                @formula(q + endog(p) ~ pc(2) + 0),  # Extract 2 PCs
                :id, :t, :S;
                guess=[guess],
                quiet=true,
                complete_coverage=false,
                tol=1e-7,
                pca_option=(; impute_method=:zero, demean=false, maxiter=100, algorithm=DeflatedHeteroPCA(t_block=10), suppress_warnings=true, abstol=1e-8),
                algorithm=:iv
            )

            push!(convergence_pc, model_pc.converged ? 1 : 0)
            if model_pc.converged
                push!(results_pc_extraction, endog_coef(model_pc)[1])
            else
                push!(results_pc_extraction, NaN)
            end

        catch e
            push!(convergence_pc, 0)
            push!(results_pc_extraction, NaN)
        end

        # try
        #     # Method 3: PC extraction with StandardHeteroPCA
        #     model_pc_standard = giv(
        #         df,
        #         @formula(q + endog(p) ~ pc(2) + 0),  # Extract 2 PCs
        #         :id, :t, :S;
        #         guess=[guess],
        #         quiet=true,
        #         complete_coverage=false,
        #         tol=1e-7,
        #         pca_option=(; impute_method=:zero, demean=false, maxiter=100, algorithm=StandardHeteroPCA(), suppress_warnings=true, abstol=1e-8),
        #         algorithm=:iv
        #     )

        #     push!(convergence_pc_standard, model_pc_standard.converged ? 1 : 0)
        #     if model_pc_standard.converged
        #         push!(results_pc_standard, endog_coef(model_pc_standard)[1])
        #     else
        #         push!(results_pc_standard, NaN)
        #     end

        # catch e
        #     push!(convergence_pc_standard, 0)
        #     push!(results_pc_standard, NaN)
        # end

        # try
        #     # Method 4: PC extraction with DiagonalDeletion
        #     model_pc_diagonal = giv(
        #         df,
        #         @formula(q + endog(p) ~ pc(2) + 0),  # Extract 2 PCs
        #         :id, :t, :S;
        #         guess=[guess],
        #         quiet=true,
        #         complete_coverage=false,
        #         tol=1e-7,
        #         pca_option=(; impute_method=:zero, demean=false, maxiter=100, algorithm=DiagonalDeletion(), suppress_warnings=true, abstol=1e-8),
        #         algorithm=:iv
        #     )

        #     push!(convergence_pc_diagonal, model_pc_diagonal.converged ? 1 : 0)
        #     if model_pc_diagonal.converged
        #         push!(results_pc_diagonal, endog_coef(model_pc_diagonal)[1])
        #     else
        #         push!(results_pc_diagonal, NaN)
        #     end

        # catch e
        #     push!(convergence_pc_diagonal, 0)
        #     push!(results_pc_diagonal, NaN)
        # end

        try
            # Method 5: No factor control
            model_no = giv(
                df,
                @formula(q + endog(p) ~ 0),  # No factor controls
                :id, :t, :S;
                guess=[guess],
                quiet=true,
                complete_coverage=false,
                algorithm=:iv
            )

            push!(convergence_no, model_no.converged ? 1 : 0)
            if model_no.converged
                push!(results_no_factors, endog_coef(model_no)[1])
            else
                push!(results_no_factors, NaN)
            end

        catch e
            push!(convergence_no, 0)
            push!(results_no_factors, NaN)
        end

        try
            # Method 6: Standard GIV
            result = estimate_standard_giv(df, 2)
            elasticity = -result.coef[2]

            push!(convergence_standard_giv, 1)
            push!(results_standard_giv, elasticity)

        catch e
            push!(convergence_standard_giv, 0)
            push!(results_standard_giv, NaN)
        end
    end
    
    # Filter out NaN values for analysis
    valid_true = results_true_factors[.!isnan.(results_true_factors)]
    valid_pc = results_pc_extraction[.!isnan.(results_pc_extraction)]
    # valid_pc_standard = results_pc_standard[.!isnan.(results_pc_standard)]
    # valid_pc_diagonal = results_pc_diagonal[.!isnan.(results_pc_diagonal)]
    valid_no = results_no_factors[.!isnan.(results_no_factors)]
    valid_standard_giv = results_standard_giv[.!isnan.(results_standard_giv)]

    # True elasticity value
    true_elasticity = 1.0 / simparams.M  # Should be 2.0

    # Calculate summary statistics
    println("\n=== SIMULATION RESULTS ===")
    println("True elasticity: $(true_elasticity)")
    println()

    println("Method 1 - True Factors:")
    println("  Convergence rate: $(mean(convergence_true)*100)%")
    if length(valid_true) > 0
        println("  Mean estimate: $(round(mean(valid_true), digits=3))")
        println("  Bias: $(round(mean(valid_true) - true_elasticity, digits=3))")
        println("  Std deviation: $(round(std(valid_true), digits=3))")
        println("  RMSE: $(round(sqrt(mean((valid_true .- true_elasticity).^2)), digits=3))")
    end
    println()

    println("Method 2 - PC Extraction (Deflated):")
    println("  Convergence rate: $(mean(convergence_pc)*100)%")
    if length(valid_pc) > 0
        println("  Mean estimate: $(round(mean(valid_pc), digits=3))")
        println("  Bias: $(round(mean(valid_pc) - true_elasticity, digits=3))")
        println("  Std deviation: $(round(std(valid_pc), digits=3))")
        println("  RMSE: $(round(sqrt(mean((valid_pc .- true_elasticity).^2)), digits=3))")
    end
    println()

    # println("Method 3 - PC Extraction (Standard):")
    # println("  Convergence rate: $(mean(convergence_pc_standard)*100)%")
    # if length(valid_pc_standard) > 0
    #     println("  Mean estimate: $(round(mean(valid_pc_standard), digits=3))")
    #     println("  Bias: $(round(mean(valid_pc_standard) - true_elasticity, digits=3))")
    #     println("  Std deviation: $(round(std(valid_pc_standard), digits=3))")
    #     println("  RMSE: $(round(sqrt(mean((valid_pc_standard .- true_elasticity).^2)), digits=3))")
    # end
    println()

    # println("Method 4 - PC Extraction (Diagonal Deletion):")
    # println("  Convergence rate: $(mean(convergence_pc_diagonal)*100)%")
    # if length(valid_pc_diagonal) > 0
    #     println("  Mean estimate: $(round(mean(valid_pc_diagonal), digits=3))")
    #     println("  Bias: $(round(mean(valid_pc_diagonal) - true_elasticity, digits=3))")
    #     println("  Std deviation: $(round(std(valid_pc_diagonal), digits=3))")
    #     println("  RMSE: $(round(sqrt(mean((valid_pc_diagonal .- true_elasticity).^2)), digits=3))")
    # end
    # println()

    println("Method 5 - No Factor Control:")
    println("  Convergence rate: $(mean(convergence_no)*100)%")
    if length(valid_no) > 0
        println("  Mean estimate: $(round(mean(valid_no), digits=3))")
        println("  Bias: $(round(mean(valid_no) - true_elasticity, digits=3))")
        println("  Std deviation: $(round(std(valid_no), digits=3))")
        println("  RMSE: $(round(sqrt(mean((valid_no .- true_elasticity).^2)), digits=3))")
    end
    println()

    println("Method 6 - Standard GIV:")
    println("  Convergence rate: $(mean(convergence_standard_giv)*100)%")
    if length(valid_standard_giv) > 0
        println("  Mean estimate: $(round(mean(valid_standard_giv), digits=3))")
        println("  Bias: $(round(mean(valid_standard_giv) - true_elasticity, digits=3))")
        println("  Std deviation: $(round(std(valid_standard_giv), digits=3))")
        println("  RMSE: $(round(sqrt(mean((valid_standard_giv .- true_elasticity).^2)), digits=3))")
    end
    println()

    # Basic tests
    @testset "Convergence rates" begin
        @test mean(convergence_true) > 0.7  # At least 70% convergence
        @test mean(convergence_pc) > 0.5    # PC method might be less stable
        @test mean(convergence_pc_standard) > 0.5    # Standard PC method
        # @test mean(convergence_pc_diagonal) > 0.5    # Diagonal deletion method
        @test mean(convergence_no) > 0.5    # No factors might have issues
        @test mean(convergence_standard_giv) > 0.5    # Standard GIV method
    end

    @testset "Bias comparison" begin
        if length(valid_true) > 5 && length(valid_pc) > 5 && length(valid_no) > 5
            bias_true = abs(mean(valid_true) - true_elasticity)
            bias_pc = abs(mean(valid_pc) - true_elasticity)
            # bias_pc_standard = length(valid_pc_standard) > 5 ? abs(mean(valid_pc_standard) - true_elasticity) : NaN
            # bias_pc_diagonal = length(valid_pc_diagonal) > 5 ? abs(mean(valid_pc_diagonal) - true_elasticity) : NaN
            bias_no = abs(mean(valid_no) - true_elasticity)
            bias_standard_giv = length(valid_standard_giv) > 5 ? abs(mean(valid_standard_giv) - true_elasticity) : NaN

            # True factors should have lowest bias
            @test bias_true <= bias_pc + 0.5  # Allow some tolerance
            @test bias_true <= bias_no + 0.5

            println("Bias comparison:")
            println("  True=$(round(bias_true,digits=3))")
            println("  PC Deflated=$(round(bias_pc,digits=3))")
            # println("  PC Standard=$(isnan(bias_pc_standard) ? "N/A" : round(bias_pc_standard,digits=3))")
            # println("  PC Diagonal=$(isnan(bias_pc_diagonal) ? "N/A" : round(bias_pc_diagonal,digits=3))")
            println("  None=$(round(bias_no,digits=3))")
            println("  Standard GIV=$(isnan(bias_standard_giv) ? "N/A" : round(bias_standard_giv,digits=3))")
        end
    end
    
    # @testset "RMSE comparison" begin
    #     if length(valid_true) > 5 && length(valid_pc) > 5 && length(valid_no) > 5
    #         rmse_true = sqrt(mean((valid_true .- true_elasticity) .^ 2))
    #         rmse_pc = sqrt(mean((valid_pc .- true_elasticity) .^ 2))
    #         rmse_pc_standard = length(valid_pc_standard) > 5 ? sqrt(mean((valid_pc_standard .- true_elasticity) .^ 2)) : NaN
    #         # rmse_pc_diagonal = length(valid_pc_diagonal) > 5 ? sqrt(mean((valid_pc_diagonal .- true_elasticity) .^ 2)) : NaN
    #         rmse_no = sqrt(mean((valid_no .- true_elasticity) .^ 2))

    #         # True factors should have lowest RMSE
    #         @test rmse_true <= rmse_pc + 0.5  # Allow some tolerance
    #         @test rmse_true <= rmse_no + 0.5

    #         println("RMSE comparison:")
    #         println("  True=$(round(rmse_true,digits=3))")
    #         println("  PC Deflated=$(round(rmse_pc,digits=3))")
    #         println("  PC Standard=$(isnan(rmse_pc_standard) ? "N/A" : round(rmse_pc_standard,digits=3))")
    #         # println("  PC Diagonal=$(isnan(rmse_pc_diagonal) ? "N/A" : round(rmse_pc_diagonal,digits=3))")
    #         println("  None=$(round(rmse_no,digits=3))")
    #     end
    # end
end