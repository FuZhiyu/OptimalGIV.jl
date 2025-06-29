using Test
using OptimalGIV
using DataFrames, Random, Statistics
using OptimalGIV: simulate_data

@testset "PC Extraction Simulation Tests" begin
    Random.seed!(123)
    
    @testset "PC extraction with homogeneous elasticity" begin
        # Simulate data with homogeneous elasticity and 2 factors
        simparams = (
            N=15,         # Number of entities (smaller for speed)
            T=50,         # Number of time periods (smaller for speed)
            K=2,          # Number of common factors
            M=0.5,        # Aggregate elasticity
            σζ=0.0,       # Homogeneous elasticity (no heterogeneity)
            σp=1.5,       # Price volatility
            σᵤcurv=0.1,
            h=0.2,        # Excess HHI
            ushare=0.5,   # Share of idiosyncratic shocks
            missingperc=0.0  # No missing data
        )
        
        # Generate single simulation
        simulated_dfs = simulate_data(simparams, Nsims=1, seed=100)
        df = simulated_dfs[1]
        
        # Convert id to string for consistency
        df.id = string.(df.id)
        sort!(df, [:t, :id])
        
        # Estimate with observed factors (baseline)
        model_observed = giv(
            df,
            @formula(q + endog(p) ~ 0 + id & (η1 + η2)),
            :id, :t, :S;
            guess=[2.0],  # Single elasticity since homogeneous
            quiet=true,
            algorithm=:iv_twopass,
        )
        
        @test model_observed.converged
        @test length(endog_coef(model_observed)) == 1
        
        # Estimate with PC extraction
        model_pc = giv(
            df,
            @formula(q + endog(p) ~ pc(2) + 0),  # Extract 2 PCs as controls
            :id, :t, :S;
            guess=[2.0],
            algorithm=:iv_twopass,
            quiet=true,
        )
        
        @test model_pc.converged
        @test length(endog_coef(model_pc)) == 1
        @test model_pc.n_pcs == 2
        
        # Estimate without PC extraction for comparison
        model_nopc = giv(
            df,
            @formula(q + endog(p) ~ 0),
            :id, :t, :S;
            guess=[2.0],
            algorithm=:iv_twopass,
            quiet=true,
        )
        
        @test model_nopc.converged
        @test model_nopc.n_pcs == 0
        
        # Compare performance: PC extraction should be closer to observed factors
        est_observed = endog_coef(model_observed)[1]
        est_pc = endog_coef(model_pc)[1]
        est_nopc = endog_coef(model_nopc)[1]
        true_elasticity = 1/simparams.M  # True aggregate elasticity
        
        @test abs(est_observed - true_elasticity) < 0.8
        @test abs(est_pc - true_elasticity) < 1.0
        @test abs(est_nopc - true_elasticity) < 1.5
        
        # PC extraction should be closer to observed factors than no PC
        @test abs(est_pc - est_observed) < abs(est_nopc - est_observed)
        
        println("True elasticity: $(round(true_elasticity, digits=3))")
        println("Observed factors: $(round(est_observed, digits=3))")
        println("PC extraction: $(round(est_pc, digits=3))")
        println("No PC: $(round(est_nopc, digits=3))")
    end
    
    @testset "PC extraction with heterogeneous elasticity" begin
        # Simulate data with heterogeneous elasticity and 1 factor
        simparams = (
            N=12,         # Number of entities
            T=40,         # Number of time periods
            K=1,          # Number of common factors
            M=0.4,        # Aggregate elasticity
            σζ=0.3,       # Heterogeneous elasticity
            σp=1.0,       # Price volatility
            σᵤcurv=0.15,
            h=0.15,       # Excess HHI
            ushare=0.6,   # Share of idiosyncratic shocks
            missingperc=0.0  # No missing data
        )
        
        # Generate single simulation
        simulated_dfs = simulate_data(simparams, Nsims=1, seed=200)
        df = simulated_dfs[1]
        
        # Convert id to string for consistency
        df.id = string.(df.id)
        sort!(df, [:t, :id])
        
        # Estimate with observed factors
        model_observed = giv(
            df,
            @formula(q + id & endog(p) ~ 0 + id & η1),  # Entity-specific elasticities
            :id, :t, :S;
            guess=ones(simparams.N) * 2.5,  # Vector of initial guesses
            quiet=true,
            algorithm=:iv_twopass,
        )
        
        @test model_observed.converged
        @test length(endog_coef(model_observed)) == simparams.N
        
        # Estimate with PC extraction
        model_pc = giv(
            df,
            @formula(q + id & endog(p) ~ pc(1) + 0),  # Extract 1 PC
            :id, :t, :S;
            guess=ones(simparams.N) * 2.5,
            algorithm=:iv_twopass,
            quiet=true,
        )
        
        @test model_pc.converged
        @test length(endog_coef(model_pc)) == simparams.N
        @test model_pc.n_pcs == 1
        
        # Estimate without PC extraction
        model_nopc = giv(
            df,
            @formula(q + id & endog(p) ~ 0),
            :id, :t, :S;
            guess=ones(simparams.N) * 2.5,
            algorithm=:iv_twopass,
            quiet=true,
        )
        
        @test model_nopc.converged
        @test model_nopc.n_pcs == 0
        
        # Compare aggregate elasticity estimates
        agg_observed = model_observed.agg_coef
        agg_pc = model_pc.agg_coef
        agg_nopc = model_nopc.agg_coef
        true_agg = 1/simparams.M
        
        # All should be reasonably close to true aggregate
        @test abs(mean(agg_observed) - true_agg) < 1.0
        @test abs(mean(agg_pc) - true_agg) < 1.5
        @test abs(mean(agg_nopc) - true_agg) < 2.0
        
        println("True aggregate elasticity: $(round(true_agg, digits=3))")
        println("Observed factors (mean): $(round(mean(agg_observed), digits=3))")
        println("PC extraction (mean): $(round(mean(agg_pc), digits=3))")
        println("No PC (mean): $(round(mean(agg_nopc), digits=3))")
    end
    
    @testset "PC extraction error handling" begin
        # Test with small dataset where PC extraction might fail
        simparams = (
            N=5,          # Very small N
            T=10,         # Very small T
            K=1,
            M=0.5,
            σζ=0.0,
            σp=1.0,
            σᵤcurv=0.1,
            h=0.1,
            ushare=0.5,
            missingperc=0.0
        )
        
        simulated_dfs = simulate_data(simparams, Nsims=1, seed=300)
        df = simulated_dfs[1]
        df.id = string.(df.id)
        sort!(df, [:t, :id])
        
        # Try to extract more PCs than feasible
        @test_nowarn begin
            model = giv(
                df,
                @formula(q + endog(p) ~ pc(3) + 0),  # Request 3 PCs with only 5 entities
                :id, :t, :S;
                guess=[2.0],
                algorithm=:iv_twopass,
                quiet=true,
            )
        end
        
        # Test with :iv algorithm (should throw error)
        @test_throws ArgumentError giv(
            df,
            @formula(q + endog(p) ~ pc(1) + 0),
            :id, :t, :S;
            guess=[2.0],
            algorithm=:iv,  # Should not support PC extraction
            quiet=true,
        )
        
        # Test with :debiased_ols algorithm (should throw error)
        @test_throws ArgumentError giv(
            df,
            @formula(q + endog(p) ~ pc(1) + 0),
            :id, :t, :S;
            guess=[2.0],
            algorithm=:debiased_ols,  # Should not support PC extraction
            quiet=true,
        )
    end
end