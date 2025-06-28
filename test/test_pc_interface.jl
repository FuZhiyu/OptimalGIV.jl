using Test, OptimalGIV, DataFrames, StatsModels
using OptimalGIV: PCTerm, pc, has_pc, get_pc_k, remove_pc_terms, separate_giv_ols_fe_formulas

@testset "PC Interface Tests" begin
    
    @testset "PCTerm construction and properties" begin
        # Test pc() function creates correct PCTerm
        pc3 = pc(3)
        @test pc3 isa PCTerm{3}
        
        pc5 = pc(5)
        @test pc5 isa PCTerm{5}
        
        # Test StatsModels interface
        @test StatsModels.terms(pc3) == []
        @test StatsModels.termvars(pc3) == []
    end
    
    @testset "has_pc detection" begin
        # Test has_pc for different term types
        @test has_pc(pc(3)) == true
        @test has_pc(term(:x)) == false
        @test has_pc(ConstantTerm(1)) == false
        
        # Test formula with pc terms
        formula_with_pc = @formula(y ~ x + pc(3))
        @test has_pc(formula_with_pc) == true
        
        formula_without_pc = @formula(y ~ x + z)
        @test has_pc(formula_without_pc) == false
        
        # Test tuple detection
        rhs_terms = (term(:x), pc(2), term(:z))
        @test has_pc(rhs_terms) == true
        
        rhs_terms_no_pc = (term(:x), term(:z))
        @test has_pc(rhs_terms_no_pc) == false
    end
    
    @testset "get_pc_k extraction" begin
        # Test getting k from PCTerm
        @test get_pc_k(pc(3)) == 3
        @test get_pc_k(pc(7)) == 7
        
        # Test formula with pc terms
        formula_pc3 = @formula(y ~ x + pc(3))
        @test get_pc_k(formula_pc3) == 3
        
        formula_no_pc = @formula(y ~ x + z)
        @test get_pc_k(formula_no_pc) == 0
        
        # Test multiple pc terms (should return maximum)
        rhs_terms_multi = (term(:x), pc(2), pc(5), term(:z))
        @test get_pc_k(rhs_terms_multi) == 5
    end
    
    @testset "remove_pc_terms functionality" begin
        # Test removing PC terms from formula
        formula_with_pc = @formula(y ~ x + pc(3) + z)
        formula_no_pc = remove_pc_terms(formula_with_pc)
        
        @test !has_pc(formula_no_pc)
        
        # Check that other terms are preserved
        rhs_termvars = StatsModels.termvars(formula_no_pc.rhs)
        @test :x in rhs_termvars
        @test :z in rhs_termvars
        
        # Test formula with only PC term - should become intercept-only (y ~ 1)
        formula_only_pc = @formula(y ~ pc(3))
        formula_intercept = remove_pc_terms(formula_only_pc)
        @test formula_intercept.rhs isa InterceptTerm{true}
        @test string(formula_intercept) == "y ~ 1"
        
        # Test explicit no-intercept with PC should preserve no-intercept
        formula_no_int_pc = @formula(y ~ 0 + pc(3))
        formula_no_int = remove_pc_terms(formula_no_int_pc)
        @test formula_no_int.rhs isa ConstantTerm{Int64}
        @test formula_no_int.rhs.n == 0
        @test string(formula_no_int) == "y ~ 0"
    end
    
    @testset "Integration with separate_giv_ols_fe_formulas" begin
        # Create test data
        df = DataFrame(
            id = repeat(1:5, outer=10),
            t = repeat(1:10, inner=5),
            S = rand(50),
            q = rand(50),
            p = repeat(rand(10), inner=5),
            x = rand(50)
        )
        
        # Test formula with PC terms
        formula_with_pc = @formula(q + endog(p) ~ x + pc(3))
        
        # This should not throw an error and should return n_pcs = 3
        result = separate_giv_ols_fe_formulas(df, formula_with_pc)
        @test length(result) == 6  # Should now return 6 elements including n_pcs
        
        formula_givcore, formula_schema, fes, feids, fekeys, n_pcs = result
        @test n_pcs == 3
        
        # Check that PC terms are removed from formula_schema
        @test !has_pc(formula_schema)
        
        # Test formula without PC terms
        formula_no_pc = @formula(q + endog(p) ~ x)
        result_no_pc = separate_giv_ols_fe_formulas(df, formula_no_pc)
        formula_givcore_no_pc, formula_schema_no_pc, fes_no_pc, feids_no_pc, fekeys_no_pc, n_pcs_no_pc = result_no_pc
        @test n_pcs_no_pc == 0
    end
    
    @testset "Error handling and edge cases" begin
        # Test invalid k values - should work at construction but might fail later
        @test pc(0) isa PCTerm{0}
        @test pc(1) isa PCTerm{1}
        
        # Test very large k values
        @test pc(100) isa PCTerm{100}
        
        # Test formula parsing with complex expressions
        df = DataFrame(
            id = repeat(1:3, outer=5),
            t = repeat(1:5, inner=3),
            S = rand(15),
            q = rand(15),
            p = repeat(rand(5), inner=3),
            x1 = rand(15),
            x2 = rand(15)
        )
        
        # Multiple variables with PC
        formula_complex = @formula(q + id & endog(p) ~ x1 + x2 + pc(2))
        result_complex = separate_giv_ols_fe_formulas(df, formula_complex)
        @test length(result_complex) == 6
        @test result_complex[6] == 2  # n_pcs should be 2
    end
end