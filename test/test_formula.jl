using Test, GIV
using OptimalGIV:
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
##============= test parsing_formula =============##
response_term, slope_terms, endog_term, exog_terms = parse_endog(f)
@test response_term == GIV.term(:q)
@test slope_terms == tuple(term(:id))
response_term, slope_terms, endog_term, exog_terms = parse_endog(@formula(q + id & endog(p) ~ id & η + S & η))
@test exog_terms == tuple(InteractionTerm((term(:id), term(:η))), InteractionTerm((term(:S),  term(:η))))

_, slope_terms, _ = parse_endog(@formula(q + endog(p) ~ 0))
@test slope_terms == tuple(ConstantTerm(1))

# formula has to have endogenous variable
@test_throws ArgumentError parse_endog(@formula(q + id ~ 0))
# endogenous variables only appears on the left hand side
@test_throws ArgumentError parse_endog(@formula(q + id & endog(p) ~ endog(p)))
# only one endogenous variable is allowed
@test_throws ArgumentError parse_endog(@formula(q + id & endog(p) + endog(f) ~ 0))
# only one response variable is allowed
@test_throws ArgumentError parse_endog(@formula(q + g + id & endog(p) ~ 0))

