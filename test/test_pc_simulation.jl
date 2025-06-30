using Test, OptimalGIV, Random, Statistics, DataFrames
using OptimalGIV: simulate_data, evaluation_metrics

@testset "PC Factor Recovery Simulation" begin
    Random.seed!(123)
    
    # Simple simulation parameters for homogeneous elasticities
    simparams = (
        N=15,         # Number of entities
        T=100,         # Number of time periods  
        K=2,          # Number of common factors
        M=0.5,        # Aggregate elasticity (1/M = 2.0 true elasticity)
        σζ=0.0,       # Homogeneous elasticity (no heterogeneity)
        σp=2,       # Price volatility
        σᵤcurv=0.1,   # Size-dependent shock curvature
        h=0.2,        # Excess HHI
        ushare=0.5,   # Share of idiosyncratic shocks
        missingperc=0.1  # No missing data
    )

    # Number of simulations
    Nsims = 50

    println("Running $Nsims simulations comparing factor control methods...")

    # Storage for results
    results_true_factors = Float64[]
    results_pc_extraction = Float64[]
    results_no_factors = Float64[]

    convergence_true = Int[]
    convergence_pc = Int[]
    convergence_no = Int[]

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

        try
            # Method 1: True factors (baseline)
            model_true = giv(
                df,
                @formula(q + endog(p) ~ 0 + id & (η1 + η2)),
                :id, :t, :S;
                guess=[true_elasticity],
                quiet=true,
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
                guess=[true_elasticity],
                quiet=true,
                pca_option=(; impute_method=:pairwise, demean=true, maxiter=1000),
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

        try
            # Method 3: No factor control
            model_no = giv(
                df,
                @formula(q + endog(p) ~ 0),  # No factor controls
                :id, :t, :S;
                guess=[true_elasticity],
                quiet=true,
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
    end
    
    # Filter out NaN values for analysis
    valid_true = results_true_factors[.!isnan.(results_true_factors)]
    valid_pc = results_pc_extraction[.!isnan.(results_pc_extraction)]
    valid_no = results_no_factors[.!isnan.(results_no_factors)]

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

    println("Method 2 - PC Extraction:")
    println("  Convergence rate: $(mean(convergence_pc)*100)%")
    if length(valid_pc) > 0
        println("  Mean estimate: $(round(mean(valid_pc), digits=3))")
        println("  Bias: $(round(mean(valid_pc) - true_elasticity, digits=3))")
        println("  Std deviation: $(round(std(valid_pc), digits=3))")
        println("  RMSE: $(round(sqrt(mean((valid_pc .- true_elasticity).^2)), digits=3))")
    end
    println()

    println("Method 3 - No Factor Control:")
    println("  Convergence rate: $(mean(convergence_no)*100)%")
    if length(valid_no) > 0
        println("  Mean estimate: $(round(mean(valid_no), digits=3))")
        println("  Bias: $(round(mean(valid_no) - true_elasticity, digits=3))")
        println("  Std deviation: $(round(std(valid_no), digits=3))")
        println("  RMSE: $(round(sqrt(mean((valid_no .- true_elasticity).^2)), digits=3))")
    end
    println()

    # Basic tests
    @testset "Convergence rates" begin
        @test mean(convergence_true) > 0.7  # At least 70% convergence
        @test mean(convergence_pc) > 0.5    # PC method might be less stable
        @test mean(convergence_no) > 0.5    # No factors might have issues
    end

    @testset "Bias comparison" begin
        if length(valid_true) > 5 && length(valid_pc) > 5 && length(valid_no) > 5
            bias_true = abs(mean(valid_true) - true_elasticity)
            bias_pc = abs(mean(valid_pc) - true_elasticity)
            bias_no = abs(mean(valid_no) - true_elasticity)

            # True factors should have lowest bias
            @test bias_true <= bias_pc + 0.5  # Allow some tolerance
            @test bias_true <= bias_no + 0.5

            println("Bias comparison: True=$(round(bias_true,digits=3)), PC=$(round(bias_pc,digits=3)), None=$(round(bias_no,digits=3))")
        end
    end
    
    @testset "RMSE comparison" begin
        if length(valid_true) > 5 && length(valid_pc) > 5 && length(valid_no) > 5
            rmse_true = sqrt(mean((valid_true .- true_elasticity) .^ 2))
            rmse_pc = sqrt(mean((valid_pc .- true_elasticity) .^ 2))
            rmse_no = sqrt(mean((valid_no .- true_elasticity) .^ 2))

            # True factors should have lowest RMSE
            @test rmse_true <= rmse_pc + 0.5  # Allow some tolerance
            @test rmse_true <= rmse_no + 0.5

            println("RMSE comparison: True=$(round(rmse_true,digits=3)), PC=$(round(rmse_pc,digits=3)), None=$(round(rmse_no,digits=3))")
        end
    end
end