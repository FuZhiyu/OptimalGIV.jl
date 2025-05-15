using Test, GIV
using GIV:
    parse_giv_formula,
    InteractionTerm,
    ConstantTerm,
    term,
    has_endog,
    parse_endog,
    replace_function_term,
    EndogenousTerm

f = @formula(q + id & endog(p) ~ 0)
##============= test has_endog =============##
@test has_endog(f)
@test !has_endog(f.rhs)

##============= test replace_function_term =============##
@test replace_function_term(term(:id)) == term(:id)
@test replace_function_term(f.lhs[2]) == InteractionTerm((term(:id), EndogenousTerm(term(:p))))
@test replace_function_term(f) == GIV.FormulaTerm(
    tuple(term(:q), InteractionTerm((term(:id), EndogenousTerm(term(:p))))),
    ConstantTerm(0),
)
##============= test parsing_giv_formula =============##
f = replace_function_term(f)
f_giv, f_main = parse_giv_formula(f)
@test f_giv.lhs == GIV.term(:q)
slope_terms, endog_term = parse_endog(f_giv)
@test slope_terms == tuple(term(:id))
@test endog_term == EndogenousTerm(term(:p))

# always order the response term first
f_giv, f_main = parse_giv_formula(@formula(id & endog(p) + q ~ 0))
@test f_main.lhs[1] == term(:q)
# formula has to have endogenous variable
@test_throws ArgumentError parse_giv_formula(@formula(q + id ~ 0))
# endogenous variables only appears on the left hand side
@test_throws ArgumentError parse_giv_formula(@formula(q + id & endog(p) ~ endog(p)))
# only one response variable is allowed
@test_throws ArgumentError parse_giv_formula(@formula(q + g + id & endog(p) ~ 0))

