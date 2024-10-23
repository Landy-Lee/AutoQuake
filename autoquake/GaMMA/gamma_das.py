#%%
import pandas as pd
from gamma.utils import association, estimate_eps
import numpy as np
import os
from pyproj import Proj
from pathlib import Path
import logging

def config2csv(config, filename):
    with open(filename, 'w') as f:
        for key, value in config.items():
            f.write(f'{key},{value}\n')  
# Function to extract desired substring
def extract_substring(s):
    return s.split('.')[1]

def config_setting(vel_model, result_path, station_csv, picks_csv):

    if not result_path.exists():
        result_path.mkdir(parents=True)

    ## read picks
    picks = pd.read_csv(picks_csv, parse_dates=["phase_time"])
    picks['station_id'] = picks['station_id'].apply(extract_substring)
    picks.rename(columns={"station_id": "id", "phase_time": "timestamp", "phase_type": "type", "phase_score": "prob"}, inplace=True)
    print("Pick format:", picks.iloc[:10])

    ## read stations
    stations = pd.read_csv(station_csv)
    stations.rename(columns={"station": "id"}, inplace=True)
    print("Station format:", stations.iloc[:10])

    ## Automatic region; you can also specify a region
    x0 = stations["longitude"].mean()
    y0 = stations["latitude"].mean()
    degree2km = 111.32
    config = {
        'center': (x0, y0), 
        'xlim_degree': [x0-0.5, x0+0.5], 
        'ylim_degree': [y0-0.5, y0+0.5], 
        'degree2km': degree2km}

    ## projection to km
    proj = Proj(f"+proj=sterea +lon_0={config['center'][0]} +lat_0={config['center'][1]} +units=km")
    stations[["x(km)", "y(km)"]] = stations.apply(lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1)
    stations["z(km)"] = stations["elevation"].apply(lambda x: -x/1e3)

    ### setting GMMA configs
    config["use_amplitude"] = False

    config["method"] = "BGMM"  
    if config["method"] == "BGMM": ## BayesianGaussianMixture
        config["oversample_factor"] = 5
    if config["method"] == "GMM": ## GaussianMixture
        config["oversample_factor"] = 1
    #!!Prior for (time and amplitude): set a larger value to prevent splitting events. A too large value will cause merging events.
    # config["covariance_prior"] = [1000, 1000] 

    # earthquake location
    config["vel"] = {"p": 6.0, "s": 6.0 / 1.75}
    config["dims"] = ['x(km)', 'y(km)', 'z(km)']
    config["x(km)"] = (np.array(config["xlim_degree"])-np.array(config["center"][0]))*config["degree2km"]*np.cos(np.deg2rad(config["center"][1]))
    config["y(km)"] = (np.array(config["ylim_degree"])-np.array(config["center"][1]))*config["degree2km"]
    config["z(km)"] = (0, 60)
    config["bfgs_bounds"] = (
        (config["x(km)"][0] - 1, config["x(km)"][1] + 1),  # x
        (config["y(km)"][0] - 1, config["y(km)"][1] + 1),  # y
        (0, config["z(km)"][1] + 1),  # z
        (None, None),  # t
    )

    ## using Eikonal for 1D velocity model
    velocity_model = pd.read_csv(vel_model, names=["zz", "vp", "vs"])
    velocity_model = velocity_model[velocity_model["zz"] <= config["z(km)"][1]]
    vel = {"z": velocity_model["zz"].values, "p": velocity_model["vp"].values, "s": velocity_model["vs"].values}
    h = 1.0
    config["eikonal"] = {"vel": vel, "h": h, "xlim": config["x(km)"], "ylim": config["y(km)"], "zlim": config["z(km)"]}

    # DBSCAN: 
    ##!!Truncate the picks into segments: change the dbscan_eps to balance speed and event splitting. A larger eps prevent spliting events but can take longer time in the preprocessing step.
    config["use_dbscan"] = False
    config["dbscan_eps"] = estimate_eps(stations, config["vel"]["p"]) 
    config["dbscan_min_samples"] = 66 # this should change

    # set number of cpus
    config["ncpu"] = 35

    ##!!Post filtering (indepent of gmm): change these parameters to filter out associted picks with large errors
    station_num = stations.shape[0]
    config["min_picks_per_eq"] = station_num // 10 
    config["min_p_picks_per_eq"] = 62 # 290 = Hole A + Hole B
    config["min_s_picks_per_eq"] = 21 # 1/3
    config["max_sigma11"] = 2.0 # second
    # config["max_sigma22"] = 1.0 # amplitude
    # config["max_sigma12"] = 1.0 # covariance

    if config["use_amplitude"]:
        picks = picks[picks["amp"] != -1]

    for k, v in config.items():
        print(f"{k}: {v}")
    config2csv(config, filename= result_path / 'config.csv')
    return picks, stations, config

def run_gamma(vel_model, result_path, station_csv, picks_csv):
    # config setting
    picks, stations, config = config_setting(vel_model, result_path, station_csv, picks_csv)

    # Association with GaMMA
    event_idx0 = 0 ## current earthquake index
    assignments = []
    events, assignments = association(picks, stations, config, event_idx0, config["method"])
    event_idx0 += len(events)

    ## create catalog
    events = pd.DataFrame(events)
    events[["longitude","latitude"]] = events.apply(lambda x: pd.Series(proj(longitude=x["x(km)"], latitude=x["y(km)"], inverse=True)), axis=1)
    events["depth_km"] = events["z(km)"]
    events.to_csv(result_path /"gamma_events.csv", index=False, 
                    float_format="%.3f",
                    date_format='%Y-%m-%dT%H:%M:%S.%f')

    ## add assignment to picks
    assignments = pd.DataFrame(assignments, columns=["pick_index", "event_index", "gamma_score"])
    picks = picks.join(assignments.set_index("pick_index")).fillna(-1).astype({'event_index': int})
    picks.rename(columns={"id": "station_id", "timestamp": "phase_time", "type": "phase_type", "prob": "phase_score", "amp": "phase_amplitude"}, inplace=True)
    picks.to_csv(result_path / "gamma_picks.csv", index=False, 
                    date_format='%Y-%m-%dT%H:%M:%S.%f')
    
if __name__ == "__main__":
    vel_model = Path("/home/patrick/Work/AutoQuake/vel_model/midas_vel.csv")
    root_dir = Path("/home/patrick/Work/EQNet/tests/hualien_0403/")
    result_path = root_dir / "gamma_test" / "test_4"
    station_csv = root_dir / "station_das.csv"
    picks_csv = root_dir / "picks_phasenet_das"  / "old_shit" / "MiDAS_20240403_0_86100.csv"
    run_gamma(
        vel_model=vel_model,
        result_path=result_path,
        station_csv=station_csv,
        picks_csv=picks_csv
        )
# %%
