"""
pc_extraction.jl

Utilities for principal component extraction from panel data residuals.
"""

using HeteroPCA


"""
    extract_pcs_from_residuals(residuals, obs_index, n_pcs; <keyword arguments>)

Extract principal components from residuals and return factors, loadings, model, and updated residuals.

# Arguments
- `residuals`: Vector of residuals in stacked format
- `obs_index`: ObservationIndex containing entity/time mapping  
- `n_pcs`: Number of principal components to extract

# Keyword Arguments
- Additional keyword arguments are passed to HeteroPCA.heteropca()

# Returns
- `factors`: k×T matrix of time factors
- `loadings`: N×k matrix of entity loadings
- `pca_model`: HeteroPCAModel object
- `updated_residuals`: Vector of residuals with PC components removed
"""
function extract_pcs_from_residuals(residuals, obs_index, n_pcs; kwargs...)
    # Convert to matrix format
    residual_matrix = vector_to_matrix(residuals, obs_index)
    
    # Extract PCs using HeteroPCA with provided options
    default_options = (impute_method=:zero, demean=false, maxiter=100, algorithm=DeflatedHeteroPCA(t_block=10))
    pca_options = merge(default_options, kwargs)
    pca_model = heteropca(residual_matrix, n_pcs; pca_options...)
    
    # Get factors and reconstructed matrix
    factors_kt = predict(pca_model, residual_matrix)  # k×T matrix
    reconstructed = HeteroPCA.reconstruct(pca_model, factors_kt)  # N×T matrix
    
    # Format outputs
    factors = factors_kt  # Keep as k×T matrix (native HeteroPCA format)
    loadings_matrix = loadings(pca_model)  # N×k matrix (loadings: projection * sqrt(PC vars))
    
    # Remove PC components from residuals
    pc_residuals = residual_matrix - reconstructed
    # Convert back to stacked format
    updated_residuals = matrix_to_vector(pc_residuals, obs_index)
    
    return factors, loadings_matrix, pca_model, updated_residuals
end

