using Test, OptimalGIV, DataFrames, Random, LinearAlgebra
using OptimalGIV: simulate_data, build_error_function, nested_pc_solve
using OptimalGIV.HeteroPCA: DeflatedHeteroPCA

# Nested PC solver: outer PC extraction / inner fixed-quadratic ζ solve with the
# analytic Jacobian, opt-in via `pc_solver = :nested` for internal-PC specs.
@testset "Nested PC solver" begin
    pcaopt = (; impute_method=:zero, demean=false, maxiter=200,
        algorithm=DeflatedHeteroPCA(t_block=10), suppress_warnings=true, abstol=1e-8)
    # Homogeneous single-elasticity pc(2) fixture (true ζ = 1/M = 2), the config
    # from the removed pc_two_root_analysis.jl where both solvers converge.
    simparams = (N=40, T=400, K=2, M=0.5, σζ=0.0, σp=2.0, σᵤcurv=0.1, h=0.2, ushare=0.5, missingperc=0.0)
    df = simulate_data(simparams; seed=7, Nsims=1)[1]
    df.id = string.(df.id)
    f = @formula(q + endog(p) ~ 0 + pc(2))

    @testset "nested ≈ one-step at a shared fixed point" begin
        m_os = giv(df, f, :id, :t, :S; algorithm=:iv, precision_mode=:proxy, guess=[1.0],
            quiet=true, pca_option=pcaopt, pc_solver=:onestep, tol=1e-8, iterations=300)
        m_ne = giv(df, f, :id, :t, :S; algorithm=:iv, precision_mode=:proxy, guess=[1.0],
            quiet=true, pca_option=pcaopt, pc_solver=:nested, tol=1e-8, iterations=300)
        @test m_os.converged
        @test m_ne.converged
        # both target the joint fixed point g(ζ; Λ(ζ)) = 0; agree up to the PCA
        # heuristic's finite tolerance.
        @test isapprox(endog_coef(m_os)[1], endog_coef(m_ne)[1]; atol=5e-3)
    end

    @testset "default is :onestep; :nested is opt-in; validation" begin
        m_def = giv(df, f, :id, :t, :S; algorithm=:iv, precision_mode=:proxy, guess=[1.0],
            quiet=true, pca_option=pcaopt, tol=1e-8)
        m_os = giv(df, f, :id, :t, :S; algorithm=:iv, precision_mode=:proxy, guess=[1.0],
            quiet=true, pca_option=pcaopt, pc_solver=:onestep, tol=1e-8)
        # default path is byte-identical to explicit :onestep (existing behavior unchanged)
        @test endog_coef(m_def)[1] == endog_coef(m_os)[1]
        @test_throws ArgumentError giv(df, f, :id, :t, :S; algorithm=:iv,
            precision_mode=:proxy, guess=[1.0], quiet=true, pc_solver=:bogus)
    end

    @testset "nested_pc_solve trace, joint residual, guards" begin
        _, elem = build_error_function(df, f, :id, :t, :S; algorithm=:iv,
            precision_mode=:proxy, pca_option=pcaopt, quiet=true)
        ζ, conv, tr = nested_pc_solve([1.0], elem.uq, elem.uCp, elem.C, elem.S,
            elem.obs_index, true, Val(:iv); precision=elem.precision, n_pcs=elem.n_pcs,
            pca_option=pcaopt, solver_options=(; ftol=1e-8, iterations=300))
        @test conv
        @test tr.moment_resid < 1e-6            # joint residual certified below moment_tol
        @test tr.outer_iters ≥ 1
        @test length(tr.inner_iters) == tr.outer_iters
        @test tr.total_inner == sum(tr.inner_iters)
        # requires fixed precisions and n_pcs > 0
        @test_throws ArgumentError nested_pc_solve([1.0], elem.uq, elem.uCp, elem.C, elem.S,
            elem.obs_index, true, Val(:iv); precision=nothing, n_pcs=2)
        @test_throws ArgumentError nested_pc_solve([1.0], elem.uq, elem.uCp, elem.C, elem.S,
            elem.obs_index, true, Val(:iv); precision=elem.precision, n_pcs=0)
    end
end
