using Test, OptimalGIV, DataFrames, CategoricalArrays
using OptimalGIV: preprocess_dataframe
df = DataFrame(
    id=categorical(repeat(1:10, outer=50)),
    t=repeat(1:50, inner=10),
    S=rand(10 * 50),
    q=rand(10 * 50),
    p=repeat(rand(50), inner=10),
    ζ=rand(10 * 50),
    u=rand(10 * 50),
    η=rand(10 * 50)
)
df = DataFrames.shuffle(df)


##============= test preprocess_dataframe =============##
#TODO: to be fixed
df2 = preprocess_dataframe(df, @formula(q + id & endog(p) ~ η), :id, :t, :S)
@test issorted(df2, [:t, :id])


##============= test get_coefnames =============##
# df_processed = preprocess_dataframe(df, @formula(q + id & endog(p) ~ η), :id, :t, :S)
# @test slope_names == ["id: $i" for i in 1:10]
# @test factor_names == ["η"]
# @test endog_name == "q"
# @test resp_name == "p"

# slope_names, factor_names, endog_name, resp_name = OptimalGIV.get_coefnames(df_processed, @formula(q + id & endog(p) ~ id & η + S & η))
# @test factor_names == vcat(["id: $i & η" for i in 1:10], ["S & η"])

# slope_names, _ = OptimalGIV.get_coefnames(df_processed, @formula(p + endog(q) ~ η))
# @test slope_names == ["Constant"]


##============= test create_coef_dataframe =============##
id_endog_coef = rand(10)
df = DataFrame(
    id=categorical(repeat(1:10, outer=50)),
    t=categorical(repeat(1:50, inner=10)),
    S=rand(10 * 50),
    q=rand(10 * 50),
    p=repeat(rand(50), inner=10),
    ζ=rand(10 * 50),
    u=rand(10 * 50),
    η=rand(10 * 50),
    η2=rand(10 * 50),
)
values(df.t)
df.lowerS = categorical(df.S .< 0.5)
f = @formula(q + id & endog(p) ~ 0)
# df = preprocess_dataframe(df, f, :id, :t, :S)
coefdf = create_coef_dataframe(df, f, id_endog_coef, ["$id & p" for id in 1:10], :id)
@test Matrix(coefdf) == [1:10 id_endog_coef ["$id & p" for id in 1:10]]


# f = @formula(q + id & endog(p) + S & endog(p) ~ 0)
# coefdf = create_coef_dataframe(df, f, [id_endog_coef; 2.0], :id)
# @test Matrix(coefdf) == [1:10 id_endog_coef 2.0 * ones(10)]

# coefdf = create_coef_dataframe(df, @formula(q + endog(p) ~ η), [1.0; 3.0], :id)
# @test Matrix(coefdf) == float([1:10 ones(10) ones(10) * 3.0])
