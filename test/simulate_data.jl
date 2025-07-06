using OptimalGIV, DataFrames, CSV, Random
using OptimalGIV: SimModel, simulate_data

"""
    format_simparamstr(simparams)

Convert simulation parameters to a string format suitable for directory names.
"""
function format_simparamstr(simparams)
    join(["$k=$v" for (k, v) in pairs(simparams)], "_")
end

"""
    simulate_data_and_save(simparams; Nsims=1000, path="simulations", seed=1)

Generate simulation data and save to CSV files.
"""
function simulate_data_and_save(simparams; Nsims=1000, path="simulations", seed=1)
    simparamstr = format_simparamstr(simparams)
    folderpath = joinpath(path, simparamstr)
    mkpath(folderpath)

    @info "Generating $Nsims simulations for $simparamstr"
    simdata = simulate_data(simparams; Nsims=Nsims, seed=seed)

    for i in 1:Nsims
        CSV.write(joinpath(folderpath, "simdata_$i.csv"), simdata[i])
    end

    @info "Saved simulations to $folderpath"
end

# Define all simulation scenarios
const SIMULATION_SCENARIOS = [
    # Baseline model
    (label="baseline",
        params=(N=10, T=100, K=2, ushare=0.5, σζ=1.0, missingperc=0.0),
        nsims=1000),

    # Baseline model with 10% missing values
    (label="10% missing",
        params=(N=10, T=100, K=2, ushare=0.5, σζ=1.0, missingperc=0.1),
        nsims=1000),

    # Baseline model with 50% missing values
    (label="50% missing",
        params=(N=10, T=200, K=2, ushare=0.5, σζ=1.0, missingperc=0.5),
        nsims=1000),

    # 50% missing with long panels
    (label="50% missing long panel",
        params=(N=10, T=1000, K=2, ushare=0.5, σζ=1.0, missingperc=0.5),
        nsims=200),

    # Wider panel with uniform elasticity
    (label="sparse panel",
        params=(N=100, T=100, K=2, ushare=0.5, σζ=0.0, missingperc=0.9),
        nsims=400),

    # Larger panel with uniform elasticity
    (label="large panel",
        params=(N=100, T=1000, K=2, ushare=0.5, σζ=0.0, missingperc=0.9),
        nsims=400),

    # PC extraction scenarios with homogeneous elasticity
    (label="homogeneous_default",
        params=(N=10, T=100, K=2, M=0.5, σζ=0.0, σp=2.0, h=0.2, ushare=0.3, missingperc=0.0),
        nsims=400),

    (label="homogeneous_large",
        params=(N=40, T=400, K=2, M=0.5, σζ=0.0, σp=2.0, h=0.2, ushare=0.3, missingperc=0.0),
        nsims=400),

    (label="homogeneous_large_missing",
        params=(N=40, T=400, K=2, M=0.5, σζ=0.0, σp=2.0, h=0.2, ushare=0.3, missingperc=0.2),
        nsims=400)
]

"""
    generate_all_simulations()

Generate all simulation scenarios if they don't already exist.
"""
function generate_all_simulations()
    # Generate simulation parameter CSV for reference
    simparamdf = DataFrame(
        simulated_model=[s.label for s in SIMULATION_SCENARIOS],
        simparamstr=[format_simparamstr(s.params) for s in SIMULATION_SCENARIOS]
    )
    CSV.write("simulations/simparamstr.csv", simparamdf)

    # Generate each simulation set
    for scenario in SIMULATION_SCENARIOS
        simulate_data_and_save(scenario.params; Nsims=scenario.nsims)
    end
end

"""
    generate_selective_simulations(scenario_labels::Vector{String})

Generate only specified simulation scenarios by label, preserving existing simparamstr.csv.
"""
function generate_selective_simulations(scenario_labels::Vector{String})
    # Generate simulation parameter CSV for reference
    simparamdf = DataFrame(
        simulated_model=[s.label for s in SIMULATION_SCENARIOS],
        simparamstr=[format_simparamstr(s.params) for s in SIMULATION_SCENARIOS]
    )
    
    # Only update the CSV if it doesn't exist, otherwise preserve existing entries
    if !isfile("simulations/simparamstr.csv")
        CSV.write("simulations/simparamstr.csv", simparamdf)
    else
        # Read existing and append new scenarios if not present
        existing_df = CSV.read("simulations/simparamstr.csv", DataFrame)
        
        for scenario_label in scenario_labels
            if !(scenario_label in existing_df.simulated_model)
                scenario_row = filter(row -> row.label == scenario_label, SIMULATION_SCENARIOS)[1]
                new_row = DataFrame(
                    simulated_model=[scenario_label],
                    simparamstr=[format_simparamstr(scenario_row.params)]
                )
                append!(existing_df, new_row)
            end
        end
        CSV.write("simulations/simparamstr.csv", existing_df)
    end

    # Generate only the specified scenarios
    scenarios_to_generate = filter(s -> s.label in scenario_labels, SIMULATION_SCENARIOS)
    for scenario in scenarios_to_generate
        simulate_data_and_save(scenario.params; Nsims=scenario.nsims)
    end
end

"""
    generate_pc_simulations_only()

Generate only the PC extraction scenarios (last 3 scenarios) to avoid overwriting existing simulations.
"""
function generate_pc_simulations_only()
    pc_scenarios = ["homogeneous_default", "homogeneous_large", "homogeneous_large_missing"]
    generate_selective_simulations(pc_scenarios)
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    generate_pc_simulations_only()  # Only generate PC scenarios to avoid overwriting
end
