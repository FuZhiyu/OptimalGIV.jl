using GIV, DataFrames, CSV, Random
using GIV: SimModel
function format_simparamstr(simparams)
    join(["$k=$v" for (k, v) in pairs(simparams)], "_")
end

function simulate_data_and_save(simparams; Nsims=1000, path="simulations", seed=1)
    simparamstr = format_simparamstr(simparams)
    folderpath = joinpath(path, simparamstr)
    mkpath(folderpath)
    simdata = simulate_data(simparams; Nsims=Nsims, seed=seed)
    for i in 1:Nsims
        CSV.write(joinpath(folderpath, "simdata_$i.csv"), simdata[i])
    end
end

##==================== baseline model ====================
simparams = (N=10, T=100, K=2, ushare=0.5, σζ=1.0, missingperc=0.0)
simulate_data_and_save(simparams)

##==================== baseline model with 10% missing values ====================
simparams_10missing = (N=10, T=100, K=2, ushare=0.5, σζ=1.0, missingperc=0.1)
simulate_data_and_save(simparams_10missing)

##==================== baseline model with 50% missing values ====================
simparams_50missing = (N=10, T=200, K=2, ushare=0.5, σζ=1.0, missingperc=0.5)
simulate_data_and_save(simparams_50missing)

##==================== 50% missing with long panels ====================
simparams_50missing_longpanel = (N=10, T=1000, K=2, ushare=0.5, σζ=1.0, missingperc=0.5)
simulate_data_and_save(simparams_50missing_longpanel, Nsims=200)

##==================== wider panel uniform elasticity ====================
simparams_sparsepanel = (N=100, T=100, K=2, ushare=0.5, σζ=0.0, missingperc=0.9)
simulate_data_and_save(simparams_sparsepanel, Nsims=400)
##==================== larger panel uniform elasticity ====================
simparams_largepanel = (N=100, T=1000, K=2, ushare=0.5, σζ=0.0, missingperc=0.9)
simulate_data_and_save(simparams_largepanel, Nsims=400)

simparamdf = DataFrame([
        "baseline" format_simparamstr(simparams);
        "10% missing" format_simparamstr(simparams_10missing);
        "50% missing" format_simparamstr(simparams_50missing);
        "50% missing long panel" format_simparamstr(simparams_50missing_longpanel);
        "sparse panel" format_simparamstr(simparams_sparsepanel);
        "large panel" format_simparamstr(simparams_largepanel)
    ], [:simulated_model, :simparamstr])
CSV.write("simulations/simparamstr.csv", simparamdf)
