"""
heteropca_utils.jl

Utility functions for performing heterogeneous PCA analysis on panel data.
Contains functions to residualize variables and extract principal components.
"""

using DataFrames, DataFramesMeta
using HeteroPCA
using FixedEffectModels
using Statistics


"""
    extract_matrix_from_dataframe(df, entity_id, time_var, value_var)

Convert a long-format dataframe to a matrix with entities as rows and time periods as columns.

# Arguments
- `df`: DataFrame in long format
- `entity_id`: Symbol for entity identifier column
- `time_var`: Symbol for time variable column  
- `value_var`: Symbol for value variable column

# Returns
- `time_vec`: Vector of unique time periods
- `entity_vec`: Vector of unique entity identifiers
- `matrix`: Matrix with entities as rows, time as columns
"""
function extract_matrix_from_dataframe(df, entity_id, time_var, value_var)
    tmp = select(df, entity_id, time_var, value_var)
    time_vec = unique(sort(tmp[!, time_var]))
    wide = unstack(tmp, entity_id, time_var, value_var)
    sort!(wide, entity_id)
    matrix = select(wide, string.(time_vec)...) |> Matrix
    entity_vec = wide[!, entity_id]
    return time_vec, entity_vec, matrix
end

"""
    matrix_to_dataframe(time_vec, entity_vec, matrix, entity_id, time_var, value_var)

Convert a matrix back to long-format dataframe - the reverse of extract_matrix_from_dataframe.

# Arguments
- `time_vec`: Vector of time periods (corresponds to columns)
- `entity_vec`: Vector of entity identifiers (corresponds to rows)
- `matrix`: Matrix with entities as rows, time as columns
- `entity_id`: Symbol for entity identifier column name in output
- `time_var`: Symbol for time variable column name in output
- `value_var`: Symbol for value variable column name in output

# Returns
- DataFrame in long format, sorted by time_var then entity_id
"""
function matrix_to_dataframe(time_vec, entity_vec, matrix; idvar=:id, timevar=:t, valuevar=:value)
    # Get dimensions
    n_entities, n_times = size(matrix)

    # Create long format using repeat (already sorted by time then entity)
    df_long = DataFrame()
    df_long[!, idvar] = repeat(entity_vec, outer=n_times)
    df_long[!, timevar] = repeat(time_vec, inner=n_entities)
    df_long[!, valuevar] = vec(matrix)
    df_long = dropmissing(df_long)
    sort!(df_long, [timevar, idvar])
    return df_long
end



"""
    extract_pca_factors(df, y_var, entity_id, time_var, share_var; k=3, demean=false, impute_method=:zero)

Extract principal components from panel data using heterogeneous PCA.

# Arguments
- `df`: DataFrame containing the panel data
- `y_var`: Symbol for the variable to extract factors from
- `entity_id`: Symbol for entity identifier column
- `time_var`: Symbol for time variable column
- `share_var`: Symbol for share/weight variable (e.g., market share)
- `k`: Number of principal components to extract (default: 3)
- `demean`: Whether to demean the data (default: false)
- `impute_method`: Method for handling missing values (default: :zero)

# Returns
- DataFrame containing:
  - Time periods
  - k principal component factors
  - Aggregate factor loading (z-score)
"""
function extract_pca_factors(df, y_var, entity_id, time_var; k=3, demean=false, impute_method=:zero)
    # Extract matrices using helper function
    time_vec, entity_vec, y_matrix = extract_matrix_from_dataframe(df, entity_id, time_var, y_var)

    # Fit heterogeneous PCA
    pca_model = heteropca(y_matrix, k, demean=demean, impute_method=impute_method)

    # Extract factors
    factor_matrix = predict(pca_model, y_matrix)

    # Calculate factor residuals
    factor_residuals = y_matrix - reconstruct(pca_model, factor_matrix)
    residual_df = matrix_to_dataframe(time_vec, entity_vec, factor_residuals, idvar=entity_id, timevar=time_var, valuevar=Symbol("$(y_var)_res"))
    # Calculate aggregate factor loading (weighted sum of residuals)

    # Create output DataFrame
    factor_names = ["pc$(k)_$(y_var)_$(i)" for i in 1:k]
    factor_df = DataFrame(factor_matrix', factor_names)
    factor_df[!, time_var] = time_vec

    return factor_df, residual_df, pca_model
end



