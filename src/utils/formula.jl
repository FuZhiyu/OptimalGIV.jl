eachterm(@nospecialize(x::AbstractTerm)) = (x,)
eachterm(@nospecialize(x::NTuple{N,AbstractTerm})) where {N} = x

const ENDOG_CONTEXT = Any
struct GIVSlopeModel <: StatisticalModel end

struct EndogenousTerm{T} <: AbstractTerm
    x::T
end
StatsModels.terms(t::EndogenousTerm) = [t.x]
StatsModels.termvars(t::EndogenousTerm) = [Symbol(t.x)]
endog(x::Symbol) = EndogenousTerm(term(x))
endog(x::Term) = EndogenousTerm(x)

has_endog(::EndogenousTerm) = true
has_endog(::FunctionTerm{typeof(endog)}) = true
has_endog(@nospecialize(t::InteractionTerm)) = any(has_endog(x) for x in t.terms)
has_endog(::AbstractTerm) = false
has_endog(f::Tuple) = any(has_endog(x) for x in f)
has_endog(@nospecialize(t::FormulaTerm)) = any(has_endog(x) for x in eachterm(t.lhs))

endogsymbol(t::EndogenousTerm) = Symbol(t.x)
endogsymbol(t::FunctionTerm{typeof(endog)}) = Symbol(t.args[1])

separate_slope_from_endog(t::FunctionTerm{typeof(endog)}) = (ConstantTerm(1), t)
separate_slope_from_endog(t::EndogenousTerm) = (ConstantTerm(1), t)

function separate_slope_from_endog(@nospecialize(t::InteractionTerm))
    slopeterms = filter(!has_endog, t.terms)
    endog_terms = filter(has_endog, t.terms)
    if length(endog_terms) > 1
        throw(ArgumentError("Interaction term contains more than one endogenous terms"))
        return slopeterms[1], endog_terms[1]
    end
    if length(slopeterms) > 1
        # throw(ArgumentError("Double-interaction with the endogenous term is not supported yet."))
        return InteractionTerm(slopeterms), endog_terms[1]
    else
        return slopeterms[1], endog_terms[1]
    end
end

has_categorical(t::AbstractTerm) = false
has_categorical(t::CategoricalTerm) = true
has_categorical(t::InteractionTerm) = any(has_categorical(x) for x in t.terms)

replace_function_term(@nospecialize(t::FunctionTerm{typeof(endog)})) = EndogenousTerm(t.args[1])
replace_function_term(t::AbstractTerm) = t
replace_function_term(@nospecialize(t::InteractionTerm)) =
    InteractionTerm(replace_function_term.(t.terms))
replace_function_term(@nospecialize(t::Tuple)) = replace_function_term.(t)
replace_function_term(@nospecialize(t::FormulaTerm)) =
    FormulaTerm(replace_function_term(t.lhs), replace_function_term(t.rhs))

function parse_giv_formula(@nospecialize(f::FormulaTerm))
    if has_endog(f.rhs)
        throw(ArgumentError("Formula contains endogenous terms on the right-hand side"))
    end
    if has_endog(f.lhs)
        endog_terms = Tuple(term for term in eachterm(f.lhs) if has_endog(term))
        response_terms = filter(!has_endog, eachterm(f.lhs))
        if length(response_terms) > 1
            throw(ArgumentError("Formula contains more than one response term"))
        end
        response_term = response_terms[1]
        formula_giv = FormulaTerm(response_term, endog_terms + InterceptTerm{false}())
        # making sure the response term is the first term
        formula = FormulaTerm(response_term + endog_terms, f.rhs)
        return formula_giv, formula
    else
        throw(ArgumentError("Formula does not contain endogenous terms"))
    end
end

function parse_endog(@nospecialize(f::FormulaTerm))
    endog_terms = Tuple(term for term in eachterm(f.rhs) if has_endog(term))
    separated_terms = [separate_slope_from_endog(term) for term in endog_terms]
    slope_terms = Tuple(term[1] for term in separated_terms)
    endog_terms = [term[2] for term in separated_terms]
    if length(unique(endogsymbol.(endog_terms))) .> 1
        throw(ArgumentError("Formula contains more than one endogenous term"))
    end
    endog_term = endog_terms[1]
    return slope_terms, endog_term
end

function StatsModels.apply_schema(
    t::FunctionTerm{typeof(endog)},
    sch::StatsModels.Schema,
    Mod::Type{<:ENDOG_CONTEXT},
)
    return apply_schema(EndogenousTerm(t.args[1]), sch, Mod)
end

function StatsModels.apply_schema(
    t::EndogenousTerm,
    sch::StatsModels.Schema,
    Mod::Type{<:ENDOG_CONTEXT},
)
    return apply_schema(t.x, sch, Mod)
end

function StatsModels.apply_schema(
    t::EndogenousTerm,
    sch::StatsModels.Schema,
    Mod::Type{<:GIVSlopeModel}
)
    return InterceptTerm{true}()
end

"""
Adpoted from FixedEffectModels.jl
"""
function StatsModels.apply_schema(
    ft::FormulaTerm,
    sch::StatsModels.Schema,
    Mod::Type{GIVModel}, has_fe_intercept
)
    # make sure the dummies on lhs always have full rank
    schema_lhs = StatsModels.FullRank(sch)
    lhs = apply_schema(ft.lhs, schema_lhs, StatisticalModel)
    schema_rhs = StatsModels.FullRank(sch)

    if has_fe_intercept
        push!(schema_rhs.already, InterceptTerm{true}())
    end
    rhs = collect_matrix_terms(apply_schema(ft.rhs, schema_rhs, StatisticalModel))
    return FormulaTerm(lhs, rhs)
end


# ##=============== convert the formula to FixedEffectModel-compatible formula ================##
# function StatsModels.apply_schema(t::FunctionTerm{typeof(endog)}, sch::StatsModels.Schema, Mod::Type{<:_FEMODEL})
#     apply_schema(EndogenousTerm(t.args[1]), sch, Mod)
# end
# StatsModels.apply_schema(t::EndogenousTerm, schema, Mod::Type{<:_FEMODEL}) = apply_schema(t.x, schema, mod)

# function StatsModels.apply_schema(t::AbstractTerm, schema, Mod::Type{<:_ABSTRACTFEMODEL})
#     println("invoked")
#     t = apply_schema(t, schema, StatisticalModel)
#     if isa(t, CategoricalTerm)
#         t = FixedEffectModels.FixedEffectTerm(t.sym)
#     end
#     return t
# end

# function StatsModels.modelcols(e::EndogenousTerm, d::NamedTuple)
#     col = modelcols(p.)
# end
