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
df2 = preprocess_dataframe(df, @formula(q + id & endog(p) ~ η), :id, :t, :S)
@test issorted(df2, [:t, :id])

##============= test error for IDs with less than 2 observations =============##
# Create a DataFrame where some IDs have less than 2 observations
df_insufficient = DataFrame(
    id=categorical([1, 1, 2, 3, 3, 3]),  # ID 2 has only 1 observation
    t=[1, 2, 1, 1, 2, 3],
    S=[0.5, 0.5, 1.0, 0.33, 0.33, 0.34],
    q=rand(6),
    p=rand(6),
    η=rand(6)
)
# Test that it throws an ArgumentError
@test_throws ArgumentError preprocess_dataframe(df_insufficient, @formula(q + id & endog(p) ~ η), :id, :t, :S)
##============= test 
