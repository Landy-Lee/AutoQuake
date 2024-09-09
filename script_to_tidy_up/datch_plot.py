#%%
import os
import h5py
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import multiprocessing as mp
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from obspy import read, UTCDateTime
from geopy.distance import geodesic
# from collections import defaultdict
from matplotlib.ticker import MultipleLocator

normalize = lambda x: (x - np.mean(x, axis=-1, keepdims=True)) / np.std(x, axis=-1, keepdims=True)

def get_total_seconds(dt):
    return (dt - dt.normalize()).total_seconds()

def event_list_gen(catalog: Path) -> list:
    df = pd.read_csv(catalog)
    return list(df['event_index'])

def convert_channel_index(sta_name: str) -> int:
    if sta_name[:1] == 'A':
        channel_index = int(sta_name[1:])
    elif sta_name[:1] == 'B':
        channel_index = int(f"1{sta_name[1:]}")
    else:
        print("wrong format warning: please append the condition")
    return channel_index

def preprocess_csv(gamma_catalog, gamma_picks, phasenet_picks):
    '''
    Preprocessing the DataFrame for event iteration.
    '''
    df_catalog = pd.read_csv(gamma_catalog)
    df_catalog['datetime'] = pd.to_datetime(df_catalog['time'])
    df_catalog['ymd'] = df_catalog['datetime'].dt.strftime('%Y%m%d')
    df_catalog['hour'] = df_catalog['datetime'].dt.hour
    df_catalog['minute'] = df_catalog['datetime'].dt.minute
    df_catalog['seconds'] = df_catalog['datetime'].dt.second + df_catalog['datetime'].dt.microsecond / 1_000_000

    df_picks = pd.read_csv(gamma_picks)
    df_picks['phase_time'] = pd.to_datetime(df_picks['phase_time'])

    df_all_picks = pd.read_csv(phasenet_picks)
    df_all_picks['phase_time'] = pd.to_datetime(df_all_picks['phase_time'])
    df_all_picks['total_seconds'] = df_all_picks['phase_time'].apply(get_total_seconds)
    df_all_picks['system'] = df_all_picks['station_id'].apply(lambda x: str(x).split('.')[0]) # TW or MiDAS blablabla ***
    df_all_picks['station'] = df_all_picks['station_id'].apply(lambda x: str(x).split('.')[1]) # LONT (station name) or A001 (channel name)
    return df_catalog, df_picks, df_all_picks

def sta_associated(df_catalog, event_i):
    df_event = df_catalog[df_catalog['event_index'] == event_i]
    date = df_event['ymd'].iloc[0]
    event_time = df_event['time'].iloc[0] # UTCtime format
    event_total_seconds = get_total_seconds(df_event['datetime'].iloc[0])
    event_lon = df_event['longitude'].iloc[0]
    event_lat = df_event['latitude'].iloc[0]
    evt_point = (event_lat, event_lon)
    event_depth = df_event['depth_km'].iloc[0]  
    return date, event_total_seconds, event_time, event_lon, event_lat, event_depth, evt_point

def get_seis_data(date, event_time, evt_point, station_path, sac_path_parent_dir, sac_dir_name, amplify_index):
    '''
    This only retreive the seismometer waveform.
    '''
    starttime_trim = UTCDateTime(event_time) - 30
    endtime_trim = UTCDateTime(event_time) + 60
    df_station = pd.read_csv(station_path)
    df_seis_station = df_station[df_station['station'].apply(lambda x: x[1].isalpha())]
    station_list = df_seis_station['station'].to_list() # convert the dtype from object to list
    # plotting all the waveform
    station_data = {}
    sac_path = sac_path_parent_dir / date / sac_dir_name
    for sta in station_list:
        # glob the waveform
        try:
            data_path = list(sac_path.glob(f"*{sta}.*Z.*"))[0]
        except Exception as e:
            logging.info(f"we can't access the {sta}")
            continue

        sta_point = (df_station[df_station['station'] == sta]['latitude'].iloc[0], df_station[df_station['station'] == sta]['longitude'].iloc[0])
        dist = geodesic(evt_point, sta_point).km
        dist_round = np.round(dist , 1)

        # read the waveform
        st = read(data_path)
        st.taper(type='hann', max_percentage=0.05)
        st.filter("bandpass", freqmin=1, freqmax=10)
        st_check = True
        if starttime_trim < st[0].stats.starttime:
            st_check = False
        st[0].trim(starttime=starttime_trim, endtime=endtime_trim)
        sampling_rate = 1 / st[0].stats.sampling_rate
        time_sac = np.arange(0, 90+sampling_rate, sampling_rate) # using array to ensure the time length as same as time_window.
        x_len = len(time_sac)
        data_sac_raw = st[0].data / max(st[0].data) # normalize the amplitude.
        data_sac_raw = data_sac_raw * amplify_index + dist 
        # we might have the data lack in the beginning:
        if not st_check:
            data_sac = np.pad(data_sac_raw, (x_len - len(data_sac_raw), 0), mode='constant', constant_values=np.nan) # adding the Nan to ensure the data length as same as time window.
        else:    
            data_sac = np.pad(data_sac_raw, (0, x_len - len(data_sac_raw)), mode='constant', constant_values=np.nan) # adding the Nan to ensure the data length as same as time window.
        
        station_data[str(sta)] = {'time':time_sac, 'sac_data': data_sac, 'distance': dist_round}
    return station_data
def get_aso_pick_time(df_aso_picks, station_data, event_total_seconds, event_i):
    df_aso_picks = df_aso_picks[df_aso_picks['event_index'] == event_i]

    df_das_aso_picks = df_aso_picks[df_aso_picks['station_id'].apply(lambda x: x[1].isdigit())]
    df_das_aso_picks['channel_index'] = df_das_aso_picks['station_id'].apply(convert_channel_index)
    df_das_aso_picks['x'] = df_das_aso_picks['station_id'].apply(convert_channel_index)* 4.084 # 4.084 = dx
    df_das_aso_picks['y'] = df_das_aso_picks['phase_index'].apply(lambda x: x*0.01 +300 if x*0.01 - 30 < 0 else x*0.01) # 0.01 = dt

    df_seis_aso_picks = df_aso_picks[~df_aso_picks['station_id'].apply(lambda x: x[1].isdigit())]
    df_seis_aso_picks['total_seconds'] = df_seis_aso_picks['phase_time'].apply(get_total_seconds)
    df_seis_aso_picks['x'] = df_seis_aso_picks['total_seconds'] - event_total_seconds + 30 # Because we use -30 as start.
    df_seis_aso_picks['y'] = df_seis_aso_picks['station_id'].map(lambda x: station_data[x]['distance'])

    station_aso_list = list(df_aso_picks[df_aso_picks['event_index'] == event_i]['station_id'])  
    return df_seis_aso_picks, df_das_aso_picks, station_aso_list

def get_pick_time(df_all_picks, station_data, event_total_seconds):
    time_window_start = event_total_seconds - 30
    time_window_end = event_total_seconds + 60
    pick_time = df_all_picks['total_seconds'].to_numpy()
    df_event_picks = df_all_picks[(pick_time >= time_window_start) & (pick_time <= time_window_end)]
    
    df_seis_picks = df_event_picks[df_event_picks['station'].apply(lambda x: x[1].isalpha())]
    df_seis_picks['x'] = df_seis_picks['total_seconds'] - event_total_seconds + 30 # Because we use -30 as start.
    df_seis_picks['y'] = df_seis_picks['station'].map(lambda x: station_data[x]['distance'])
    
    df_das_picks = df_event_picks[~df_event_picks['station'].apply(lambda x: x[1].isalpha())]
    df_das_picks['channel_index'] = df_das_picks['station'].apply(convert_channel_index)
    df_das_picks['x'] = df_das_picks['station'].apply(convert_channel_index)* 4.084 # 4.084 = dx
    df_das_picks['y'] = df_das_picks['phase_index'].apply(lambda x: x*0.01 +300 if x*0.01 - 30 < 0 else x*0.01) # 0.01 = dt
    return df_seis_picks, df_das_picks

def interval_judge(map_min, map_max):
    if abs(map_max - map_min) <= 1.5:
        major_interval = 0.5
        minor_interval = 0.1
    else:
        major_interval = 1
        minor_interval = 0.5
    return major_interval, minor_interval

def plot_das(event_total_seconds, interval, ax, h5_parent_dir, df_das_aso_picks, df_das_picks):
    index = int(event_total_seconds // interval)
    window = [f"{interval*index}_{interval*(index+1)}.h5"]
    time_index = round(event_total_seconds % interval, 3)
    get_previous = False
    if time_index - 30 < 0 and index != 0:
        previous_window = f"{interval*(index-1)}_{interval*index}.h5"
        window.insert(0, previous_window)
        get_previous = True
    if time_index + 60 > interval and index != 287:
        next_window = f"{interval*(index+1)}_{interval*(index+2)}.h5"
        window.append(next_window)
        

    try:
        all_data = []
        for win in window:
            file = list(h5_parent_dir.glob(f"*{win}"))[0]
            if not file:
                raise IndexError(f"File not found for window {win}")
            try:
                with h5py.File(file, 'r') as fp:
                    ds = fp["data"]
                    data = ds[...] # np.array
                    dt = ds.attrs["dt_s"] # 0.01 sampling rate
                    dx = ds.attrs["dx_m"] # interval of cable ~ 4
                    nx, nt = data.shape
                    logging.info(data.shape)
                    x = np.arange(nx) * dx
                    t = np.arange(nt) * dt
                    all_data.append(data)
            except Exception as e:
                logging.info(f"Error reading {file}: {e}")
    except IndexError:
        logging.info(f"File not found for window {window}")

    # Handle the case where there is only one array in all_data
    if len(all_data) == 1:
        concatenated_data = all_data[0]
        logging.info("Only one data array, no need to concatenate.")
    elif len(all_data) > 1:
        # Concatenate all data arrays along the second axis (horizontally)
        concatenated_data = np.concatenate(all_data, axis=1)
        logging.info(f"Concatenated data shape: {concatenated_data.shape}")
    nx, nt = concatenated_data.shape
    x = np.arange(nx) * dx
    t = np.arange(nt) * dt

    ax.imshow(normalize(concatenated_data).T, cmap="seismic", vmin=-1, vmax=1, aspect="auto", extent=[x[0], x[-1], t[-1], t[0]], interpolation="none")
    if get_previous:
        ax.set_ylim(time_index + 60 + interval, time_index - 30 + interval) # because the concat, origin time should add interval.
    else:
        ax.set_ylim(time_index + 60, time_index - 30) # concat later or not would not influence the time.
    ax.scatter(
        df_das_picks[df_das_picks['phase_type'] == 'P']["channel_index"].values * dx, 
        df_das_picks[df_das_picks['phase_type'] == 'P']["phase_index"].values * dt, 
        c='r', 
        s=1, 
        alpha=0.05
        )
    ax.scatter(df_das_aso_picks[df_das_aso_picks['phase_type'] == 'P']["channel_index"].values * dx, 
               df_das_aso_picks[df_das_aso_picks['phase_type'] == 'P']["phase_index"].values * dt, 
               c='r', 
               s=1, 
               alpha=0.3
               )
    ax.scatter(df_das_picks[df_das_picks['phase_type'] == 'S']["channel_index"].values * dx, 
               df_das_picks[df_das_picks['phase_type'] == 'S']["phase_index"].values * dt, 
               c='c', 
               s=1, 
               alpha=0.05
               )
    ax.scatter(df_das_aso_picks[df_das_aso_picks['phase_type'] == 'S']["channel_index"].values * dx, 
               df_das_aso_picks[df_das_aso_picks['phase_type'] == 'S']["phase_index"].values * dt, 
               c='c', 
               s=1, 
               alpha=0.3
               )
    ax.scatter([], [], c="r", label="P")
    ax.scatter([], [], c="c", label="S")
    ax.legend()
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Time (s)")

def plot_assoc(i: int):
    logging.info(f"This is event_{i}")
    # logging.info(f"sta_associated function start")
    date, event_total_seconds, event_time, event_lon, event_lat, event_depth, evt_point = sta_associated(
        df_catalog=df_catalog,  
        event_i=i
        )
    # logging.info(f"get_seis start")
    station_data = get_seis_data(
        date=date, 
        event_time=event_time, 
        evt_point=evt_point, 
        station_path=station_path, 
        sac_path_parent_dir=sac_path_parent_dir, 
        sac_dir_name=sac_dir_name, 
        amplify_index=amplify_index
        )
    # logging.info(f"get_pick start")
    df_seis_picks, df_das_picks = get_pick_time(
        df_all_picks=df_all_picks,
        station_data=station_data, 
        event_total_seconds=event_total_seconds
        )
    # logging.info(f"get_aso start")
    df_seis_aso_picks, df_das_aso_picks, station_aso_list = get_aso_pick_time(
        df_aso_picks=df_aso_picks, 
        station_data=station_data, 
        event_total_seconds=event_total_seconds, 
        event_i=i)
    # plotting

    fig = plt.figure()
    # logging.info(f"ax1 let's go")
    ## cartopy
    map_proj = ccrs.PlateCarree()
    tick_proj = ccrs.PlateCarree()
    #f_name = "/home/patrick/.local/share/cartopy/shapefiles/natural_earth/Raster/NE1_HR_LC_SR_W.tif"
    region = [map_lon_min, map_lon_max , map_lat_min , map_lat_max]
    ax1 = fig.add_axes([0.3, 0.5, 0.4, 0.8], projection=map_proj)
    #ax1.imshow(plt.imread(f_name), origin='upper', transform=map_proj, extent=[-180, 180, -90, 90])
    ax1.coastlines()
    ax1.set_extent(region)
    ax1.add_feature(cfeature.LAND)
    ax1.add_feature(cfeature.OCEAN)
    ax1.add_feature(cfeature.COASTLINE)
    # logging.info(f"sub_ax1 let's go")
    sub_ax1 = fig.add_axes([0.3, -0.15, 0.4, 0.55], projection=map_proj)
    sub_ax1.coastlines()
    sub_ax1.set_extent([121.62, 121.64, 24.02, 24.04])
    sub_ax1.add_feature(cfeature.LAND)
    # sub_ax1.add_feature(cfeature.OCEAN)
    sub_ax1.add_feature(cfeature.COASTLINE)

    df_station = pd.read_csv(station_path)
    df_seis_station = df_station[df_station['station'].apply(lambda x: x[1].isalpha())]
    df_das_station = df_station[~df_station['station'].apply(lambda x: x[1].isalpha())]
    # plotting epicenter
    ax1.scatter(x=event_lon, y=event_lat, marker="*", color='gold', s=100, zorder=4)
    sub_ax1.scatter(x=event_lon, y=event_lat, marker="*", color='gold', s=30, zorder=4)
    # plotting all stations
    if not df_seis_station.empty:
        ax1.scatter(x=df_seis_station['longitude'], y=df_seis_station['latitude'],marker="^", color='silver', s=50, zorder=2)
        for station in df_seis_station['station']:
            if station in station_aso_list:
                ax1.scatter(x=df_seis_station[df_seis_station['station'] == station]['longitude'], y=df_seis_station[df_seis_station['station'] == station]['latitude'],marker="^", color='r',zorder=3)

    if not df_das_station.empty:
        sub_ax1.scatter(x=df_das_station['longitude'], y=df_das_station['latitude'],marker=".", color='silver', s=5, zorder=2)
        for station in df_das_station['station']:
            if station in station_aso_list:
                sub_ax1.scatter(x=df_das_station[df_das_station['station'] == station]['longitude'], y=df_das_station[df_das_station['station'] == station]['latitude'],marker=".", color='r',s=5, zorder=3)
    # map setup
    ## interval judge
    lon_major_interval, lon_minor_interval = interval_judge(map_lon_min, map_lon_max)
    lat_major_interval, lat_minor_interval = interval_judge(map_lat_min, map_lat_max)
    ## ticks boundary & format
    ax1.set_xticks(np.arange(map_lon_min, map_lon_max, lon_major_interval), crs=tick_proj)
    ax1.set_xticks(np.arange(map_lon_min, map_lon_max, lon_minor_interval), minor=True, crs=tick_proj)
    ax1.set_yticks(np.arange(map_lat_min, map_lat_max, lat_major_interval), crs=tick_proj)
    ax1.set_yticks(np.arange(map_lat_min, map_lat_max, lat_minor_interval), minor=True, crs=tick_proj)
    ax1.xaxis.set_major_formatter(LongitudeFormatter())
    ax1.yaxis.set_major_formatter(LatitudeFormatter())
    ax1.tick_params(axis='both', which='major', length=3, width=1, labelsize=5)
    # setting title
    ax1.set_title(f"{event_time} \nLon:{event_lon}, Lat:{event_lat}, Depth:{event_depth}", fontsize = 10)
    ax1.set_aspect('auto')
    ## ticks boundary & format
    sub_ax1.set_xticks(np.arange(das_lon_min, das_lon_max, 0.01), crs=tick_proj)
    # sub_ax1.set_xticks(np.arange(das_lon_min, das_lon_max, lon_minor_interval), minor=True, crs=tick_proj)
    sub_ax1.set_yticks(np.arange(das_lat_min, das_lat_max, 0.01), crs=tick_proj)
    # sub_ax1.set_yticks(np.arange(das_lat_min, das_lat_max, lat_minor_interval), minor=True, crs=tick_proj)
    sub_ax1.xaxis.set_major_formatter(LongitudeFormatter())
    sub_ax1.yaxis.set_major_formatter(LatitudeFormatter())
    sub_ax1.set_aspect('auto')
    # logging.info(f"ax2 let's go")
    ax2 = fig.add_axes([0.82, 0.5, 0.8, 0.9])
    
    ## waveform
    ### seis
    for sta in list(station_data.keys()):
        if sta not in station_aso_list:
            # plot the waveform transparent
            ax2.plot(station_data[sta]['time'], station_data[sta]['sac_data'], color='k', linewidth=0.4, alpha=0.25, zorder =1)
            ax2.text(station_data[sta]['time'][-1]+1, station_data[sta]['distance'], sta, fontsize=4, verticalalignment='center', alpha=0.25)
        else:
            ax2.plot(station_data[sta]['time'], station_data[sta]['sac_data'], color='k', linewidth=0.4, zorder =1)
            ax2.text(station_data[sta]['time'][-1]+1, station_data[sta]['distance'], sta, fontsize=4, verticalalignment='center')
            pass
    ## all pick
    ax2.scatter(
        df_seis_picks[df_seis_picks['phase_type'] == 'P']['x'], 
        df_seis_picks[df_seis_picks['phase_type'] == 'P']['y'], 
        color='r', 
        s =1, 
        zorder =2
        )
    ax2.scatter(
        df_seis_picks[df_seis_picks['phase_type'] == 'S']['x'], 
        df_seis_picks[df_seis_picks['phase_type'] == 'S']['y'], 
        color='c', 
        s =1, 
        zorder =2
        ) 
    ## aso_picks
    ax2.plot(
        [df_seis_aso_picks[df_seis_aso_picks['phase_type'] == 'P']['x'], df_seis_aso_picks[df_seis_aso_picks['phase_type'] == 'P']['x']], 
        [df_seis_aso_picks[df_seis_aso_picks['phase_type'] == 'P']['y'].to_numpy() - 2, df_seis_aso_picks[df_seis_aso_picks['phase_type'] == 'P']['y'].to_numpy() + 2], 
        color='r', 
        linewidth=0.7, 
        zorder =2
        )
    ax2.plot(
        [df_seis_aso_picks[df_seis_aso_picks['phase_type'] == 'S']['x'], df_seis_aso_picks[df_seis_aso_picks['phase_type'] == 'S']['x']], 
        [df_seis_aso_picks[df_seis_aso_picks['phase_type'] == 'S']['y'].to_numpy() - 2, df_seis_aso_picks[df_seis_aso_picks['phase_type'] == 'S']['y'].to_numpy() + 2], 
        color='c', 
        linewidth=0.7, 
        zorder =2
        ) 
    ax2.set_xlim(0,90)
    ax2.xaxis.set_minor_locator(MultipleLocator(5))
    ax2.yaxis.set_minor_locator(MultipleLocator(10))
    ax2.tick_params(axis='both', which='major', length=3, width=1, labelsize=5)
    #ax2.tick_params(axis='x', which='minor', length=4, width=0.5)
    ax2.set_xlabel("Time (s)", fontsize = 7)
    ### das
    # logging.info(f"ax3 let's go")
    ax3 = fig.add_axes([0.82, -0.15, 0.8, 0.55])
    plot_das(
        event_total_seconds=event_total_seconds, 
        interval=interval, 
        ax=ax3, 
        h5_parent_dir=h5_parent_dir, 
        df_das_aso_picks=df_das_aso_picks, 
        df_das_picks=df_das_picks
        )

    filename = f"{event_time.replace(':','_').replace('.','_')}.png"
    figure_dir = parent_dir / date / 'Figure'
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.subplots_adjust(wspace=0.01)
    plt.savefig(figure_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f'{event_time} plotting over')  


if __name__ == '__main__':
    # parent directory which contains all needed files
    parent_dir = Path('/home/patrick/Work/EQNet/tests/hualien_0403/check_fig')
    h5_parent_dir = Path('/raid4/DAS_data/iDAS_MiDAS/hdf5/20240403_hdf5/')
    phasenet_picks = Path('/home/patrick/Work/EQNet/tests/hualien_0403/picks_phasenet_das/all_20240403_picks.csv')
    gamma_catalog = Path("/home/patrick/Work/EQNet/tests/hualien_0403/gamma_seis_das/gamma_order.csv")
    gamma_picks = Path("/home/patrick/Work/EQNet/tests/hualien_0403/gamma_seis_das/gamma_picks.csv")
    station_path = Path("/home/patrick/Work/EQNet/tests/hualien_0403/station_all.csv")

    # logging config
    logging.basicConfig(
        filename=parent_dir / f'plot.log', 
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S', 
        filemode='w'
        )
    # the parent directory of sac data
    sac_path_parent_dir = Path("/home/patrick/Work/AutoQuake/Hualien0403/20240401_20240428/Dataset")
    sac_dir_name = 'data_final' # name of sac_dir, see the example directory

    # global dataframe
    df_catalog, df_aso_picks, df_all_picks = preprocess_csv(gamma_catalog, gamma_picks, phasenet_picks)    
    
    # waveform amplify index
    amplify_index = 5
    interval = 300
    # Cartopy map setting
    map_lon_min = 119.7 # 119.7 for whole Taiwan
    map_lon_max = 122.5 # 122.5 for whole Taiwan
    map_lat_min = 21.7 # 21.7 for whole Taiwan
    map_lat_max = 25.4 # 25.4 for whole Taiwan
    das_lon_min = 121.62 # 119.7 for whole Taiwan
    das_lon_max = 121.64 # 122.5 for whole Taiwan
    das_lat_min = 24.02 # 21.7 for whole Taiwan
    das_lat_max = 24.04 # 25.4 for whole Taiwan
    # mp
    event_list = event_list_gen(gamma_catalog)
    logging.info("processing!")
    # plot_assoc(500)
    with mp.Pool(processes=30) as pool:
        pool.map(plot_assoc, event_list) 
# %%
