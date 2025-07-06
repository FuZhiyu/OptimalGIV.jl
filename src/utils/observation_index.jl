"""
observation_index.jl

Utilities for managing observation indices in panel data structures.
Provides efficient mapping between entity-time pairs and observation indices,
with support for unbalanced panels and exclusion constraints.
"""

"""
    ObservationIndex

Structure to efficiently map between panel data observations and entity-time pairs.

# Fields
- `start_indices::Vector{Int}`: Starting index for each time period
- `end_indices::Vector{Int}`: Ending index for each time period  
- `ids::Vector{Int}`: Entity IDs for each observation
- `entity_obs_indices::Matrix{Int}`: N×T matrix where entry [i,t] contains the observation index of entity i in period t (0 if not present)
- `exclpairs::BitMatrix`: N×N matrix indicating which entity pairs are excluded from moment conditions
- `N::Int`: Total number of unique entities
- `T::Int`: Total number of unique time periods

# Notes
- Data must be sorted by (time, id) for efficient processing
- Supports unbalanced panels where not all entities appear in all time periods
- Entity IDs follow natural sort order for consistency with StatsModels.jl
"""
struct ObservationIndex
    start_indices::Vector{Int}      # Starting index for each time period
    end_indices::Vector{Int}        # Ending index for each time period
    ids::Vector{Int}                # Entity IDs for each observation
    entity_obs_indices::Matrix{Int} # N×T matrix: entity_obs_indices[i,t] = observation index of entity i in period t, or 0 if not present
    exclpairs::BitMatrix            # N×N matrix: (i,j) pairs that are excluded from moment conditions
    N::Int                          # Total number of entities 
    T::Int                          # Total number of time periods
end

"""
    create_observation_index(df, id, t, exclude_pairs=Dict{Int,Vector{Int}}())

Create an ObservationIndex structure from panel data.

# Arguments
- `df`: DataFrame containing panel data (will be sorted by [t, id] if not already)
- `id`: Symbol for entity identifier column
- `t`: Symbol for time identifier column
- `exclude_pairs`: Dictionary specifying entity pairs to exclude from moment conditions
                  Keys are entity IDs, values are vectors of entity IDs to exclude

# Returns
- `ObservationIndex`: Structure containing observation mappings and metadata

# Notes
- Uses natural sort order for entity IDs to ensure consistency with StatsModels.jl
- This ensures that coefficient vectors and residual variance vectors have consistent ordering
- Data will be sorted by (time, id) if not already sorted
"""
function create_observation_index(df, id, t, exclude_pairs=Dict{Int,Vector{Int}}())
    # Ensure data is sorted by (time, id)
    if !issorted(df, [t, id])
        sort!(df, [t, id])
    end

    # Use natural sort order to ensure consistency with StatsModels.jl
    # This ensures residual_variance vector aligns with coefficient ordering
    unique_ids = sort(unique(df[!, id]))
    unique_times = sort(unique(df[!, t]))
    
    N = length(unique_ids)
    T = length(unique_times)

    # Get id mapping using sorted order
    id_map = Dict(unique_ids .=> 1:N)
    time_map = Dict(unique_times .=> 1:T)

    # Create vectors to store indices
    start_indices = zeros(Int, T)
    end_indices = zeros(Int, T)
    ids = [id_map[row[id]] for row in eachrow(df)]

    # Create matrix to store observation indices for each entity in each period
    # 0 indicates entity is not present in that period
    entity_obs_indices = zeros(Int, N, T)

    # Find start and end indices for each time period
    current_time = time_map[df[1, t]]
    start_indices[current_time] = 1

    for i in 2:nrow(df)
        time_idx = time_map[df[i, t]]
        if time_idx != current_time
            end_indices[current_time] = i - 1
            start_indices[time_idx] = i
            current_time = time_idx
        end
    end
    end_indices[current_time] = nrow(df)

    # Fill the entity_obs_indices matrix
    for i in 1:nrow(df)
        entity_id = id_map[df[i, id]]
        time_id = time_map[df[i, t]]
        entity_obs_indices[entity_id, time_id] = i  # Store actual observation index
    end

    exclpairs = create_exclusion_matrix(unique(df[!, id]), exclude_pairs)

    return ObservationIndex(start_indices, end_indices, ids, entity_obs_indices, exclpairs, N, T)
end

"""
    create_exclusion_matrix(id_values, exclude_pairs)

Create a symmetric BitMatrix indicating which entity pairs should be excluded from moment conditions.

# Arguments
- `id_values`: Vector of unique entity IDs (in their original form)
- `exclude_pairs`: Dictionary where keys are entity IDs and values are vectors of entity IDs to exclude

# Returns
- `BitMatrix`: N×N symmetric matrix where entry [i,j] = true if pair (i,j) should be excluded

# Example
```julia
exclude_pairs = Dict(1 => [2, 3], 4 => [5])  # Exclude pairs (1,2), (1,3), (4,5)
exclmat = create_exclusion_matrix([1, 2, 3, 4, 5], exclude_pairs)
```
"""
function create_exclusion_matrix(id_values, exclude_pairs)
    # Create mapping from id to index
    id2ind = Dict(id_values[i] => i for i in 1:length(id_values))
    N = length(id_values)
    # Initialize exclusion matrix
    exclmat = zeros(Bool, N, N)

    # Fill in exclusions based on provided pairs
    for (id1, id2vec) in exclude_pairs
        i = id2ind[id1]
        for id2 in id2vec
            j = id2ind[id2]
            exclmat[i, j] = true
            exclmat[j, i] = true  # Make it symmetric
        end
    end

    return BitArray(exclmat)
end

"""
    vector_to_matrix(values, obs_index)

Convert stacked vector to N×T matrix using observation index.

# Arguments
- `values`: Vector of values in stacked format (length = total observations)
- `obs_index`: ObservationIndex containing entity/time mapping

# Returns
- `matrix`: N×T matrix where matrix[i,t] is the value for entity i at time t
            Missing values indicate entity i is not observed at time t

# Notes
- Efficiently handles unbalanced panels by using entity_obs_indices
- Missing observations are represented as `missing` in the output matrix
- Can be used for any panel data values (residuals, prices, quantities, etc.)
"""
function vector_to_matrix(values::Vector{F}, obs_index) where F
    N, T = obs_index.N, obs_index.T
    
    # Initialize matrix with missing values (handles unbalanced panels)
    matrix = Matrix{Union{F,Missing}}(missing, N, T)
    
    # Efficiently fill matrix using entity_obs_indices
    for i in 1:N
        for t in 1:T
            obs_idx = obs_index.entity_obs_indices[i, t]
            if obs_idx > 0  # Entity i is present in time period t
                matrix[i, t] = values[obs_idx]
            end
            # If obs_idx == 0, entity i is not present in time t, so matrix[i,t] remains missing
        end
    end
    
    return matrix
end

"""
    matrix_to_vector(matrix, obs_index)

Convert N×T matrix back to stacked vector format using observation index.

# Arguments
- `matrix`: N×T matrix of values (may contain missing values)
- `obs_index`: ObservationIndex containing entity/time mapping

# Returns
- `values`: Vector of values in stacked format (same length and order as original data)

# Notes
- Preserves the original observation order from the panel data
- Handles missing values in the matrix (they are skipped in the output)
- Can be used for any panel data values (residuals, prices, quantities, etc.)
"""
function matrix_to_vector(matrix, obs_index)
    N, T = obs_index.N, obs_index.T
    n_obs = length(obs_index.ids)
    
    # Initialize output vector
    values = zeros(eltype(matrix), n_obs)
    
    # Efficiently fill vector using entity_obs_indices
    for i in 1:N
        for t in 1:T
            obs_idx = obs_index.entity_obs_indices[i, t]
            if obs_idx > 0  # Entity i is present in time period t
                values[obs_idx] = matrix[i, t]
            end
        end
    end
    
    return values
end