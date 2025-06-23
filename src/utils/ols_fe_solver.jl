


function ols_with_fixed_effects(Y, X, fes;
    nthreads=Threads.nthreads(), maxiter=10000, tol=1e-6, progress_bar=true, double_precision=true, coefnames_Y=["$i-th LHS variable" for i in 1:size(Y, 2)], coefnames_X=["$i-th RHS variable" for i in 1:size(X, 2)])

    Y, X = copy(Y), copy(X)

    if !isempty(fes)
        cols = vcat(eachcol(Y), eachcol(X))
        sumsquares_pre = [sum(abs2, x) for x in cols]
        weights = uweights(size(Y, 1))
        # partial out fixed effects
        feM = AbstractFixedEffectSolver{double_precision ? Float64 : Float32}(fes, weights, Val{:cpu}, nthreads)

        # partial out fixed effects
        _, iterations, convergeds = solve_residuals!(cols, feM; maxiter=maxiter, tol=tol, progress_bar=progress_bar)

        # set variables that are likely to be collinear with the fixed effects to zero
        for i in 1:length(cols)
            if sum(abs2, cols[i]) < tol * sumsquares_pre[i]
                if i <= size(Y, 2)
                    # colinearity not allowed in the current implementation
                    throw(ArgumentError("$(coefnames_Y[i]) is probably perfectly explained by fixed effects."))
                else
                    throw(ArgumentError("$(coefnames_X[i-size(Y, 2)]) is collinear with the fixed effects."))
                end
            end
        end

        # convergence info
        iterations = maximum(iterations)
        converged = all(convergeds)
        if converged == false
            @info "Convergence not achieved in $(iterations) iterations; try increasing maxiter or decreasing tol."
        end
        # tss_partial = tss(y, has_intercept | has_fe_intercept, weights)
    else
        feM = nothing
    end

    coef, residuals = ols_step(Y, X)
    return Y, X, coef, residuals, feM
end

function ols_step(Y, X)
    XX = X'X
    Nx = size(X, 2)
    Ny = size(Y, 2)
    Xy = Symmetric(hvcat(2, XX, X' * Y,
        zeros(Ny, Nx), zeros(Ny, Ny)))
    invsym!(Xy; diagonal=1:Nx)
    coef = Xy[1:Nx, (Nx+1):end]
    residuals = Y - X * coef
    return coef, residuals
end


"""
    ols_with_fixed_effects(df, formula::FormulaTerm; save=:residuals, contrasts=Dict{Symbol,Any}(), nthreads=Threads.nthreads(), maxiter=10000, tol=1e-6, progress_bar=true, double_precision=true)

A wrapper function for run fixed effect estimation directly with dataframe and fomula. 
"""
function ols_with_fixed_effects(df, formula::FormulaTerm; save=:residuals, contrasts=Dict{Symbol,Any}(), nthreads=Threads.nthreads(), maxiter=10000, tol=1e-6, progress_bar=true, double_precision=true)
    if !omitsintercept(formula) & !hasintercept(formula)
        formula = FormulaTerm(formula.lhs, InterceptTerm{true}() + formula.rhs)
    end
    formula, formula_fes = parse_fe(formula)

    fes, feids, fekeys = parse_fixedeffect(df, formula_fes)
    has_fe_intercept = any(fe.interaction isa UnitWeights for fe in fes)
    if has_fe_intercept
        formula = FormulaTerm(formula.lhs, tuple(InterceptTerm{false}(), (term for term in eachterm(formula.rhs) if !isa(term, Union{ConstantTerm,InterceptTerm}))...))
    end

    s = schema(formula, df, contrasts)
    formula_schema = apply_schema(formula, s, GIVModel, has_fe_intercept)
    oldX = convert(Matrix{Float64}, modelcols(formula_schema.rhs, df))
    coefnames_X = coefnames(formula_schema.rhs)

    oldY = modelcols(collect_matrix_terms(formula_schema.lhs), df)
    coefnames_Y = coefnames(formula_schema.lhs)

    Y, X, coef, residuals, feM = ols_with_fixed_effects(oldY, oldX, fes; nthreads=nthreads, maxiter=maxiter, tol=tol, progress_bar=progress_bar, double_precision=double_precision, coefnames_Y=coefnames_Y, coefnames_X=coefnames_X)

    return Y, X, residuals, coef, formula_schema, feM, feids, fekeys, oldY, oldX, coefnames_Y, coefnames_X
end