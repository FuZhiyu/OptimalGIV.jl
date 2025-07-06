"""
pc_extraction.jl

Utilities for principal component extraction from panel data residuals.
"""

using HeteroPCA

"""
    vector_to_matrix(residuals, obs_index)

Convert stacked residuals vector to N×T matrix using observation index.

# Arguments
- `residuals`: Vector of residuals in stacked format (length = total observations)
- `obs_index`: ObservationIndex containing entity/time mapping

# Returns
- `residual_matrix`: N×T matrix where residual_matrix[i,t] is the residual for entity i at time t
"""
function vector_to_matrix(residuals::Vector{F}, obs_index) where F
    N, T = obs_index.N, obs_index.T
    
    # Initialize matrix with zeros (handles missing observations)
    residual_matrix = Matrix{Union{F,Missing}}(missing, N, T)
    
    # Efficiently fill matrix using entity_obs_indices
    for i in 1:N
        for t in 1:T
            obs_idx = obs_index.entity_obs_indices[i, t]
            if obs_idx > 0  # Entity i is present in time period t
                residual_matrix[i, t] = residuals[obs_idx]
            end
            # If obs_idx == 0, entity i is not present in time t, so residual_matrix[i,t] remains missing
        end
    end
    
    return residual_matrix
end

"""
    matrix_to_vector(residual_matrix, obs_index)

Convert N×T residual matrix back to stacked vector format using observation index.

# Arguments
- `residual_matrix`: N×T matrix of residuals
- `obs_index`: ObservationIndex containing entity/time mapping

# Returns
- `residuals`: Vector of residuals in stacked format (same length and order as original)
"""
function matrix_to_vector(residual_matrix, obs_index)
    N, T = obs_index.N, obs_index.T
    n_obs = length(obs_index.ids)
    
    # Initialize output vector
    residuals = zeros(eltype(residual_matrix), n_obs)
    
    # Efficiently fill vector using entity_obs_indices
    for i in 1:N
        for t in 1:T
            obs_idx = obs_index.entity_obs_indices[i, t]
            if obs_idx > 0  # Entity i is present in time period t
                residuals[obs_idx] = residual_matrix[i, t]
            end
        end
    end
    
    return residuals
end

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

