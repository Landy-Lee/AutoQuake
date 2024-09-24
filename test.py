#%%
import pandas as pd
import os
import glob
import h5py
import math
import logging
import calendar
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import multiprocessing as mp
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from obspy import read, UTCDateTime, Trace
from obspy.imaging.beachball import beach
from geopy.distance import geodesic
from collections import defaultdict
from pyrocko import moment_tensor as pmt
from pyrocko.plot import beachball, mpl_color
from autoquake.visualization import catalog_compare, pack, check_format, station_mask

normalize = lambda x: (x - np.mean(x, axis=-1, keepdims=True)) / np.std(x, axis=-1, keepdims=True)

# for example, L"A"TB is True, which represents seismometer, A"0"35 is False, which is DAS.
sta_type_distinguish = lambda x: str(x)[1].isalpha()

def get_total_seconds(dt):
    return (dt - dt.normalize()).total_seconds()

def quick_line_check(path):
    with open(path, 'r') as r:
        lines = r.readlines()
    counter = 0
    for i in lines:
        if i.strip()[:4] == '2024':
            counter += 1
    print(counter)

def convert_channel_index(sta_name: str) -> int:
    if sta_name[:1] == 'A':
        channel_index = int(sta_name[1:])
    elif sta_name[:1] == 'B':
        channel_index = int(f"1{sta_name[1:]}")
    else:
        print("wrong format warning: please append the condition")
    return channel_index

def degree_trans(part):
    """Transform degree-minute-second to decimal degrees."""
    if len(part) == 7:
        deg = int(part[:2])
        if part[2:4] == '  ':
            value = 0
        else:
            value = int(part[2:4]) / 60
        dig = value + int(part[5:]) / 3600
    else:
        deg = int(part[:3])
        if part[3:5] == '  ':
            value = 0
        else:
            value = int(part[3:5]) / 60
        dig = value + int(part[6:]) / 3600
    return deg+dig

def check_time(year: int, month: int, day: int, hour: int, min: int, sec: float):
    """
    Adjust time by handling overflow of minutes, hours, and days.
    
    Args:
        year (int): Year component of the time.
        month (int): Month component of the time (1-12).
        day (int): Day component of the time (depends on month).
        hour (int): Hour component of the time (0-23).
        min (int): Minute component of the time (0-59, can overflow).
        sec (float): Seconds component of the time.
    
    Returns:
        tuple: Adjusted (year, month, day, hour, min, sec) considering overflow.
    """
    
    # Handle second overflow (if needed)
    if sec >= 60:
        min += int(sec // 60)  # Increment minutes by seconds overflow
        sec = sec % 60         # Keep remaining seconds

    # Handle minute overflow
    if min >= 60:
        hour += min // 60      # Increment hours by minute overflow
        min = min % 60         # Keep remaining minutes

    # Handle hour overflow
    if hour >= 24:
        day += hour // 24      # Increment days by hour overflow
        hour = hour % 24       # Keep remaining hours

    # Handle day overflow (check if day exceeds days in the current month)
    while day > calendar.monthrange(year, month)[1]:  # Get number of days in month
        day -= calendar.monthrange(year, month)[1]    # Subtract days in current month
        month += 1                                    # Increment month

        # Handle month overflow
        if month > 12:
            month = 1
            year += 1                                 # Increment year if month overflows
    
    return year, month, day, hour, min, sec
    
def hout_generate(polarity_dout: Path, analyze_year: str)-> pd.DataFrame:
    """
    Because the CWA polarity dout did not have the corresponded hout file, so create one.
    """
    with open(polarity_dout, 'r') as r:
        lines = r.readlines()
    data = []
    for line in lines:
        if line.strip()[:4] == analyze_year:
            year = int(line[1:5].strip())
            month = int(line[5:7].strip())
            day = int(line[7:9].strip())
            hour = int(line[9:11].strip())
            min = int(line[11:13].strip())
            second = float(line[13:19].strip())
            time = f"{year:4}-{month:02}-{day:02}T{hour:02}:{min:02}:{second:05.2f}"
            lat_part = line[19:26].strip()
            lon_part = line[26:34].strip()
            event_lon = round(degree_trans(lon_part),3)
            event_lat = round(degree_trans(lat_part),3)
            depth = line[34:40].strip()
            data.append([time, event_lat, event_lon, depth])
    columns = ['time', 'latitude', 'longitude', 'depth']
    df = pd.DataFrame(data,columns=columns)
    return df

def find_index_backward(gamma_catalog: Path, focal_dict: dict) -> int:
    """
    This is used to find the gamma event that survive in comparing CWA catalog
    and gamma catalog.
    """
    df_catalog = pd.read_csv(gamma_catalog)
    catalog_dt = pd.to_datetime(df_catalog['time'])
    std_time = pd.to_datetime(focal_dict['utc_time'])
    if min(abs((catalog_dt - std_time).dt.total_seconds())) < 1:
        close_row = abs((catalog_dt - std_time).dt.total_seconds()).idxmin()
        return df_catalog.loc[close_row].event_index
    else:
        raise ValueError("Correspnded event not founded.")
    
def preprocess_gamma_csv(gamma_catalog: Path, gamma_picks: Path, 
                         event_i:int) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    '''
    Preprocessing the DataFrame for each event index.

    This function takes the Path object from catalog and picks generated 
    by GaMMA, creating the DataFrame object for later utilization.
    
    Parameters
    ----------
    gamma_catalog : Path
        The path of gamma_events.csv
    gamma_picks : Path
        The path of gamma_picks.csv

    Returns
    -------
    df_event : DataFrame
        The DataFrame object of the gamma events i's event information.
    event_dict : dict
        The dictionary contains the needed event information.
    df_event_picks : DataFrame
        The DataFrame object of the gamma events i's associated picks.
    
    '''
    df_catalog = pd.read_csv(gamma_catalog)
    df_event = df_catalog[df_catalog['event_index'] == event_i].copy()
    df_event.loc[:, 'datetime'] = pd.to_datetime(df_event['time'])
    df_event.loc[:, 'ymd'] = df_event['datetime'].dt.strftime('%Y%m%d')
    df_event.loc[:, 'hour'] = df_event['datetime'].dt.hour
    df_event.loc[:, 'minute'] = df_event['datetime'].dt.minute
    df_event.loc[:, 'seconds'] = df_event['datetime'].dt.second + df_event['datetime'].dt.microsecond / 1_000_000
    event_dict = {
        'date': df_event['ymd'].iloc[0],
        'event_time': df_event['time'].iloc[0],
        'event_total_seconds': get_total_seconds(df_event['datetime'].iloc[0]),
        'event_lat': df_event['latitude'].iloc[0],
        'event_lon': df_event['longitude'].iloc[0],
        'event_point': (df_event['latitude'].iloc[0], df_event['longitude'].iloc[0]),
        'event_depth': df_event['depth_km'].iloc[0]       
    }

    df_picks = pd.read_csv(gamma_picks)
    df_event_picks = df_picks[df_picks['event_index'] == event_i].copy()
    # TODO: does there have any possible scenario?
    # df_event_picks['station_id'] = df_event_picks['station_id'].map(lambda x: str(x).split('.')[1])
    df_event_picks.loc[:, 'phase_time'] = pd.to_datetime(df_event_picks['phase_time'])
    
    return df_event, event_dict, df_event_picks

def process_phasenet_csv(phasenet_picks_parent: Path, date: str):
    phasenet_picks = list(phasenet_picks_parent.glob(f'*{date}'))[0]
    df_all_picks = pd.read_csv(phasenet_picks / 'picks.csv')
    df_all_picks['phase_time'] = pd.to_datetime(df_all_picks['phase_time'])
    df_all_picks['total_seconds'] = df_all_picks['phase_time'].apply(get_total_seconds)
    df_all_picks['system'] = df_all_picks['station_id'].apply(lambda x: str(x).split('.')[0]) # TW or MiDAS blablabla ***
    df_all_picks['station'] = df_all_picks['station_id'].apply(lambda x: str(x).split('.')[1]) # LONT (station name) or A001 (channel name)
    return df_all_picks

def find_das_data(event_index: int, hdf5_parent_dir: Path, polarity_picks: Path,
                  interval=300, train_window=0.64, visual_window=2,
                  sampling_rate=100):
    """
    temp function, disentangle is needed.
    """
    df_pol = pd.read_csv(polarity_picks)
    df_pol.rename(columns={'station_id': 'station'}, inplace=True)
    df_pol_clean = df_pol[df_pol["event_index"] == event_index]
    df_pol_clean = df_pol_clean[station_mask(df_pol_clean)]
    das_plot_dict={}
    for _, row in df_pol_clean.iterrows():
        if row.polarity == 'U':
            polarity = '+'
        elif row.polarity == 'D':
            polarity = '-'
        else:
            polarity = ' '
        total_seconds = get_total_seconds(pd.to_datetime(row.phase_time))
        index = int(total_seconds // interval)
        window = f"{interval*index}_{interval*(index+1)}.h5"
        try:
            file = list(hdf5_parent_dir.glob(f"*{window}"))[0]
        except IndexError:
            logging.info(f"File not found for window {window}")
            
        channel_index = convert_channel_index(row.station)
        tr = Trace()
        try:
            with h5py.File(file, 'r') as fp:
                ds = fp["data"]
                data = ds[channel_index]
                tr.stats.sampling_rate = 1 / ds.attrs['dt_s']
                tr.stats.starttime = ds.attrs['begin_time']
                tr.data = data
            p_arrival = UTCDateTime(row.phase_time)
            train_starttime_trim = p_arrival - train_window
            train_endtime_trim = p_arrival + train_window
            window_starttime_trim = p_arrival - visual_window
            window_endtime_trim = p_arrival + visual_window
            
            tr.detrend('demean')
            tr.detrend('linear')
            tr.filter("bandpass", freqmin=1, freqmax=10)
            tr.taper(0.001)
            tr.resample(sampling_rate=sampling_rate)
            # this is for visualize length
            tr.trim(starttime=window_starttime_trim, endtime=window_endtime_trim)
            visual_time = np.arange(0, 2*visual_window+1/sampling_rate, 1/sampling_rate) # using array to ensure the time length as same as time_window.
            visual_sac = tr.data 
            # this is actual training length
            tr.trim(starttime=train_starttime_trim, endtime=train_endtime_trim)
            start_index = visual_window - train_window
            train_time = np.arange(start_index, start_index + 2*train_window+1/sampling_rate, 1/sampling_rate) # using array to ensure the time length as same as time_window.
            train_sac = tr.data
            # final writing
            if 'station_info' not in das_plot_dict:
                das_plot_dict['station_info'] = {}
            das_plot_dict['station_info'][row.station] = {
                'polarity': polarity, 'p_arrival': p_arrival, 'visual_time':visual_time,
                'visual_sac': visual_sac,'train_time':train_time,'train_sac': train_sac
                }
        except Exception as e:
            print(e)
    return das_plot_dict

def find_sac_data(event_time: str, date: str, event_point: tuple[float, float], 
                 station_list: Path, sac_parent_dir: Path, 
                 sac_dir_name: str, amplify_index: float, 
                 ins_type='seis')->dict:
    """
    Retrieve the waveform from SAC.

    This function glob the sac file from each station, multiplying the 
    amplify index to make the PGA/PGV more clear, then packs the 
    waveform data into dictionary for plotting.

    """
    starttime_trim = UTCDateTime(event_time) - 30
    endtime_trim = UTCDateTime(event_time) + 60
    df_station = pd.read_csv(station_list)
    if ins_type == 'seis':
        df_station = df_station[df_station['station'].apply(lambda x: x[1].isalpha())]
    #*** TODO: Adding the read hdf5 for DAS
    elif ins_type == 'DAS':
        df_station = df_station[df_station['station'].apply(lambda x: x[1].isdigit())]

    sac_path = sac_parent_dir / date / sac_dir_name
    sac_dict = {}
    for sta in df_station['station'].to_list():
        # glob the waveform
        try:
            data_path = list(sac_path.glob(f"*{sta}.*Z.*"))[0]
        except Exception as e:
            logging.info(f"we can't access the {sta}")
            continue

        sta_point = (df_station[df_station['station'] == sta]['latitude'].iloc[0], df_station[df_station['station'] == sta]['longitude'].iloc[0])
        dist = geodesic(event_point, sta_point).km
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
        try:
            data_sac_raw = st[0].data / max(st[0].data) # normalize the amplitude.
        except Exception as e:
            logging.info(f"Error: {e}")
            logging.info(f"check the length of given time: {len(st[0].data)}")
            continue
        data_sac_raw = data_sac_raw * amplify_index + dist 
        # we might have the data lack in the beginning:
        if not st_check:
            data_sac = np.pad(data_sac_raw, (x_len - len(data_sac_raw), 0), mode='constant', constant_values=np.nan) # adding the Nan to ensure the data length as same as time window.
        else:    
            data_sac = np.pad(data_sac_raw, (0, x_len - len(data_sac_raw)), mode='constant', constant_values=np.nan) # adding the Nan to ensure the data length as same as time window.
        
        sac_dict[str(sta)] = {'time':time_sac, 'sac_data': data_sac, 'distance': dist_round}
    return sac_dict   

def find_phasenet_pick(event_total_seconds: float, sac_dict: dict, 
                       df_all_picks: pd.DataFrame, first_half=30, 
                       second_half=60, dx=4.084, dt=0.01, hdf5_time=300):
    """
    Filtering waveform in specific time window and convert it for scatter plot.
    """
    time_window_start = event_total_seconds - first_half
    time_window_end = event_total_seconds + second_half
    pick_time = df_all_picks['total_seconds'].to_numpy()
    df_event_picks = df_all_picks[(pick_time >= time_window_start) & (pick_time <= time_window_end)]

    df_seis_picks = df_event_picks[df_event_picks['station'].map(sta_type_distinguish)]
    if not df_seis_picks.empty:
        df_seis_picks['x'] = df_seis_picks['total_seconds'] - event_total_seconds + first_half # Because we use -30 as start.
        df_seis_picks['y'] = df_seis_picks['station'].map(lambda x: sac_dict[x]['distance'])

    df_das_picks = df_event_picks[~df_event_picks['station'].map(sta_type_distinguish)]
    if not df_das_picks.empty:
        df_das_picks['channel_index'] = df_das_picks['station'].map(convert_channel_index)
        df_das_picks['x'] = df_das_picks['station'].map(convert_channel_index)* dx # 4.084 = dx
        # TODO: This should thinks again about continuity.
        df_das_picks['y'] = df_das_picks['phase_index'].map(lambda x: x*dt + hdf5_time if x*dt - first_half < 0 else x*dt) # 0.01 = dt


    return df_seis_picks, df_das_picks

def find_gamma_pick(df_gamma_picks: pd.DataFrame, sac_dict: dict, 
                      event_total_seconds: float, first_half=30, 
                      dx=4.084, dt=0.01, hdf5_time=300):
    """
    Preparing the associated picks for scatter plot.
    """
    df_das_aso_picks = df_gamma_picks[~df_gamma_picks['station_id'].apply(sta_type_distinguish)]
    if not df_das_aso_picks.empty:
        df_das_aso_picks['channel_index'] = df_das_aso_picks['station_id'].apply(convert_channel_index)
        df_das_aso_picks['x'] = df_das_aso_picks['station_id'].apply(convert_channel_index)* dx 
        df_das_aso_picks['y'] = df_das_aso_picks['phase_index'].apply(lambda x: x*dt + hdf5_time if x*dt - first_half < 0 else x*dt) 

    
    df_seis_aso_picks = df_gamma_picks[df_gamma_picks['station_id'].apply(sta_type_distinguish)]
    if not df_seis_aso_picks.empty:
        df_seis_aso_picks['total_seconds'] = df_seis_aso_picks['phase_time'].apply(get_total_seconds)
        df_seis_aso_picks['x'] = df_seis_aso_picks['total_seconds'] - event_total_seconds + first_half # Because we use -30 as start.
        df_seis_aso_picks['y'] = df_seis_aso_picks['station_id'].map(lambda x: sac_dict[x]['distance'])

    return df_seis_aso_picks, df_das_aso_picks

def _plot_loc_das(df_das_station, das_region, focal_dict_list, fig=None):
    map_proj = ccrs.PlateCarree() 
    tick_proj = ccrs.PlateCarree()

    if fig is None:
        fig = plt.figure()
    ax = fig.add_axes([0.1, 0.0, 0.4, 0.47])
    # plot das
    ax.scatter(
        x=df_das_station['longitude'], y=df_das_station['latitude'],
        marker=".", color='silver', s=10, zorder=2
        )
    
    color_list = ['r', 'b']
    for i, (focal_dict, color) in enumerate(zip(focal_dict_list, color_list)):
        ax.scatter(
            x=focal_dict['longitude'], y=focal_dict['latitude'],
            marker="*", color=color, s=200, alpha=0.5, zorder=4
            )
        # plot associated station
        # TODO: maybe there exists a better way, too complicated, modify this
        if i == 0:
            for sta in focal_dict['station_info'].keys():
                if sta in list(df_das_station['station']):
                    ax.scatter(
                        x=df_das_station[df_das_station['station'] == sta]['longitude'],
                        y=df_das_station[df_das_station['station'] == sta]['latitude'],
                        marker=".", color='darkorange',s=5, zorder=3
                        )

def _plot_loc_seis(df_seis_station: pd.DataFrame, seis_region: list[float],
                   focal_dict_list: list[dict], name_list: list[str],
                   fig=None, gs=None):
    """
    This is success! DAS and both not test yet.
    ---
    check_loc_compare(
        station_list=seis_station_info,
        focal_dict_list=[gamma_focal_dict, cwa_focal_dict],
        name_list=['GaMMA', 'CWA'],
        seis_region=[119.7, 122.5, 21.7, 25.4]
        )
    """
    map_proj = ccrs.PlateCarree() 
    tick_proj = ccrs.PlateCarree()

    if fig is None:
        fig = plt.figure(figsize=(36,20))
        gs = GridSpec(10, 18, left=0, right=0.9, top=0.95 , bottom=0.42 , wspace=0.1)
    ax = fig.add_subplot(gs[1:, :4], projection=map_proj)
    ax.coastlines()
    ax.set_extent(seis_region)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)

    ax.scatter(
        x=df_seis_station['longitude'], y=df_seis_station['latitude'],
        marker="^", color='silver', s=80, zorder=2
        )
    # plot the event
    color_list = ['r', 'b']
    for i, (focal_dict, name, color) in enumerate(zip(focal_dict_list, name_list, color_list)):
        ax.scatter(
            x=focal_dict['longitude'], y=focal_dict['latitude'],
            marker="*", color=color, s=300, zorder=4
            )
        ax.text(
            seis_region[0], seis_region[3]+0.05+i/6,
            f"{name} Catalog:\nEvent time: {focal_dict['utc_time']}, Lat: {focal_dict['latitude']},Lon: {focal_dict['longitude']}, Depth: {focal_dict['depth']}",
            color=color, fontsize=13, fontweight='bold'
            )
        # plot associated station
        # TODO: maybe there exists a better way
        if i == 0:
            for sta in focal_dict['station_info'].keys():
                if sta in list(df_seis_station['station']):
                    ax.scatter(
                        x=df_seis_station[df_seis_station['station'] == sta]['longitude'],
                        y=df_seis_station[df_seis_station['station'] == sta]['latitude'],
                        marker="^", color='darkorange', s=80, zorder=3
                        )

def _plot_loc_both(df_seis_station, df_das_station,
                   seis_region, das_region, focal_dict_list, fig=None):
    """
    TODO: Find the figsize for DAS+Seis
    focal_dict_list contains 2 dict, one is CWA, another is GaMMA, they both
    got event location that needs to plot on the map.
    """
    map_proj = ccrs.PlateCarree()
    tick_proj = ccrs.PlateCarree()    
    if fig is None:
        fig = plt.figure()
    ax = fig.add_axes([0.3, 0.5, 0.4, 0.8], projection=map_proj)
    # cartopy setting
    ax.coastlines()
    ax.set_extent(seis_region)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    # DAS map
    sub_ax = fig.add_axes([0.3, -0.15, 0.4, 0.55], projection=map_proj)
    sub_ax.coastlines()
    sub_ax.set_extent(das_region) 
    sub_ax.add_feature(cfeature.LAND)
    # sub_ax1.add_feature(cfeature.OCEAN)
    sub_ax.add_feature(cfeature.COASTLINE)
    
    # plot all station
    ax.scatter(
        x=df_seis_station['longitude'], y=df_seis_station['latitude'],
        marker="^", color='silver', s=50, zorder=2
        )
    sub_ax.scatter(
        x=df_das_station['longitude'], y=df_das_station['latitude'],
        marker=".", color='silver', s=5, zorder=2
        )
    
    # plot the event
    color_list = ['r', 'b']
    for i, (focal_dict, color) in enumerate(zip(focal_dict_list, color_list)):
        ax.scatter(
            x=focal_dict['longitude'], y=focal_dict['latitude'],
            marker="*", color=color, s=100, zorder=4
            )
        sub_ax.scatter(
            x=focal_dict['longitude'], y=focal_dict['latitude'],
            marker="*", color=color, s=30, zorder=4
            )
        
        # plot associated station
        # TODO: maybe there exists a better way
        if i == 0:
            for sta in focal_dict['station_info'].keys():
                if sta in list(df_seis_station['station']):
                    ax.scatter(
                        x=df_seis_station[df_seis_station['station'] == sta]['longitude'],
                        y=df_seis_station[df_seis_station['station'] == sta]['latitude'],
                        marker="^", color='darkorange',zorder=3
                        )
                elif sta in list(df_das_station['station']):
                    sub_ax.scatter(
                        x=df_das_station[df_das_station['station'] == sta]['longitude'],
                        y=df_das_station[df_das_station['station'] == sta]['latitude'],
                        marker=".", color='darkorange',s=5, zorder=3
                        )
                else:
                    raise ValueError("No this kind of station, check it.")
# plotter
def check_loc_compare(station_list, focal_dict_list, name_list,
              seis_region=None, das_region=None, fig=None):
    """
    The aim here is to plot the epicenter of these two Catalog.
    Parameters
    ----------
    region : Tuple[float, float, float, float]
        region[0]: min longitude
        region[1]: max longitude
        region[2]: min latitude
        region[3]: max latitude
    """
    # judgement
    df_station = pd.read_csv(station_list)
    df_seis_station = df_station[df_station['station'].map(sta_type_distinguish)]
    df_das_station = df_station[~df_station['station'].map(sta_type_distinguish)]
    if not df_seis_station.empty:
        if not df_das_station.empty:
            _plot_loc_both(
                df_seis_station, df_das_station, seis_region, das_region,
                focal_dict_list, fig
                )
        else:
            _plot_loc_seis(
                df_seis_station, seis_region, focal_dict_list, name_list, fig
                )
    elif not df_das_station.empty:
        _plot_loc_das(
            df_das_station, das_region, focal_dict_list, fig
            )
    
def add_event_index(target_file: Path, output_file: Path, analyze_year: str):
    """
    Adding the index after h3dd format.
    """
    event_index = 0
    with open(target_file, 'r') as r:
        lines = r.readlines()
    with open(output_file, 'w') as event:
        for line in lines:
            if line.strip()[:4] == analyze_year: 
                event_index += 1
                event.write(f"{' ':1}{line.strip()}{' ':1}{event_index:<5}\n")
            else: 
                event.write(f"{' ':1}{line.strip()}{' ':1}{event_index:<5}\n")

def add_station_focal(polarity_index_file, focal_dict, analyze_year, event_index,
                      sac_parent_dir: Path, sac_dir_name: str, source: str,
                      train_window=0.64, visual_window=2, sampling_rate=100):
    """
    This is for dout format.
    Parse azimuth, takeoff angle, and polarity from the output file.
    """
    with open(polarity_index_file, 'r') as r:
        lines = r.read().splitlines()
    for line in lines:
        if line.strip()[:4] == analyze_year and line.split()[-1] == str(event_index):
            year = int(line[:5].strip())
            month = int(line[5:7].strip())
            day = int(line[7:9].strip())
            hour = int(line[9:11].strip())
            date = f"{year}{month:>02}{day:>02}"
            # TODO Testify the event time again
        elif line.split()[-1] == str(event_index):
            sta = line[1:5].strip()
            azi = int(line[12:15].strip())
            toa = int(line[16:19].strip())
            polarity = line[19:20]
            p_min = int(line[20:23].strip())
            p_sec = float(line[23:29].strip())
            if 'station_info' not in focal_dict:
                focal_dict['station_info'] = {}
            year, month, day, hour, p_min, p_sec = check_time(
                year, month, day, hour, p_min, p_sec
            )
            p_arrival = UTCDateTime(year, month, day, hour, p_min, p_sec)
            # TODO: A better way to branch it
            if source == 'CWA':
                focal_dict['station_info'][sta] = {'p_arrival': p_arrival,
                                                   'azimuth': azi,
                                                   'takeoff_angle': toa,
                                                   'polarity': polarity
                                                   }
                continue
            # finding sac part
            # TODO: we need to create a _get_sac() & _get_das()
            sac_path = sac_parent_dir / date / sac_dir_name
            train_starttime_trim = p_arrival - train_window
            train_endtime_trim = p_arrival + train_window
            window_starttime_trim = p_arrival - visual_window
            window_endtime_trim = p_arrival + visual_window
            try:
                # TODO: filtering like using 00 but not 10.
                data_path = list(sac_path.glob(f"*{sta}.*Z.*"))[0]
            except Exception as e:
                logging.info(f"we can't access the {sta}")
                continue
            st = read(data_path)
            st.detrend('demean')
            st.detrend('linear')
            st.filter("bandpass", freqmin=1, freqmax=10)
            st.taper(0.001)
            st.resample(sampling_rate=sampling_rate)
            # this is for visualize length
            st[0].trim(starttime=window_starttime_trim, endtime=window_endtime_trim)
            visual_time = np.arange(0, 2*visual_window+1/sampling_rate, 1/sampling_rate) # using array to ensure the time length as same as time_window.
            visual_sac = st[0].data 
            # this is actual training length
            st[0].trim(starttime=train_starttime_trim, endtime=train_endtime_trim)
            start_index = visual_window - train_window
            train_time = np.arange(start_index, start_index + 2*train_window+1/sampling_rate, 1/sampling_rate) # using array to ensure the time length as same as time_window.
            train_sac = st[0].data
            # final writing
            focal_dict['station_info'][sta] = {'p_arrival': p_arrival,
                                               'azimuth': azi,
                                               'takeoff_angle': toa,
                                               'polarity': polarity,
                                               'visual_time':visual_time,
                                               'visual_sac': visual_sac,
                                               'train_time':train_time,
                                               'train_sac': train_sac
                                               }

# def add_station_focal_sac(focal_dict: dict, date: str, sac_parent_dir: Path,
#                           sac_dir_name: str,
#                           train_window=0.64, visual_window=2, sampling_rate=100):

#     station_list = set(focal_dict['station_info'].keys())
#     sac_path = sac_parent_dir / date / sac_dir_name
#     for sta in station_list:
#         focal_sac_dict = {}
#         p_arrival = focal_dict['station_info'][sta]['p_arrival']
#         train_starttime_trim = p_arrival - train_window
#         train_endtime_trim = p_arrival + train_window
#         window_starttime_trim = p_arrival - visual_window
#         window_endtime_trim = p_arrival + visual_window
#         try:
#             data_path = list(sac_path.glob(f"*{sta}.*Z.*"))[0]
#         except Exception as e:
#             logging.info(f"we can't access the {sta}")
#             continue
#         st = read(data_path)
#         st.detrend('demean')
#         st.detrend('linear')
#         st.filter("bandpass", freqmin=1, freqmax=10)
#         st.taper(0.001)
#         st.resample(sampling_rate=sampling_rate)
#         # this is for visualize length
#         st[0].trim(starttime=window_starttime_trim, endtime=window_endtime_trim)
#         visual_time = np.arange(0, 2*visual_window+1/sampling_rate, 1/sampling_rate) # using array to ensure the time length as same as time_window.
#         visual_sac = st[0].data 
#         # this is actual training length
#         st[0].trim(starttime=train_starttime_trim, endtime=train_endtime_trim)
#         start_index = visual_window - train_window
#         train_time = np.arange(start_index, start_index + 2*train_window+1/sampling_rate, 1/sampling_rate) # using array to ensure the time length as same as time_window.
#         train_sac = st[0].data 
#         # adding a marker for recognize common station between main and comp
#         focal_sac_dict[sta] = {'visual_time':visual_time,
#                                 'visual_sac': visual_sac,
#                                 'train_time':train_time,
#                                 'train_sac': train_sac
#                                 }

#         focal_dict['station_info'][sta].update(focal_sac_dict[sta])
    
def find_gafocal_polarity(gafocal_df: pd.DataFrame, polarity_file: Path, polarity_file_index: Path,
                          analyze_year: str, event_index: int, source: str, hout_file=None):
    """
    From gafocal to find the match event's polarity information and more.

    event_index: start from 0
    """
    # TODO: currently we add hout index and polarity index here, but this would change.
    if hout_file is not None:
        df_hout, _ = check_format(hout_file)
    else:
        df_hout = hout_generate(
            polarity_dout=polarity_file, analyze_year=analyze_year
        )
    df_hout["tmp_index"] = df_hout.index + 1
    df_hout['timestamp'] = df_hout["time"].apply(lambda x: datetime.fromisoformat(x).timestamp())

    gafocal_df["timestamp"] = gafocal_df["time"].apply(lambda x: datetime.fromisoformat(x).timestamp())
    
    event_info = gafocal_df.iloc[event_index]
    focal_dict = {'utc_time': event_info.time, 'timestamp': event_info.timestamp}
    focal_dict['focal_plane'] = {
        'strike': int(event_info.strike.split('+')[0]),
        'dip': int(event_info.dip.split('+')[0]),
        'rake': int(event_info.rake.split('+')[0])
        }
    focal_dict['quality_index'] = event_info.quality_index
    focal_dict['num_of_polarity'] = event_info.num_of_polarity
    df_test = df_hout[np.abs(df_hout["timestamp"] - event_info.timestamp) < 1]
    df_test = df_test[
        (np.abs(df_test["longitude"] - event_info.longitude) < 0.02) &  # Use tolerance for floats
        (np.abs(df_test["latitude"] - event_info.latitude) < 0.02)      # Use tolerance for floats
    ]

    tmp_index = df_test['tmp_index'].iloc[0]
    
    add_event_index(
        target_file=polarity_file,
        output_file=polarity_file_index,
        analyze_year=analyze_year
        )

    add_station_focal(
        polarity_index_file=polarity_file_index,
        focal_dict=focal_dict,
        analyze_year=analyze_year,
        event_index=tmp_index,
        sac_parent_dir=sac_parent_dir,
        sac_dir_name=sac_dir_name,
        source=source
    )
    return focal_dict                

def find_compared_gafocal_polarity(gafocal_df: pd.DataFrame, polarity_file: Path, polarity_file_index: Path,
                          analyze_year: str, event_index: int, use_gamma: bool, source: str,
                          hout_file=None):
    """
    From gafocal to find the match event's polarity information and more.

    event_index: event_index here is connect to CWA catalog's row index!
    """
    # TODO: currently we add hout index and polarity index here, but this would change.
    if hout_file is not None:
        df_hout, _ = check_format(hout_file)
    else:
        df_hout = hout_generate(
            polarity_dout=polarity_file, analyze_year=analyze_year
        )
    df_hout["tmp_index"] = df_hout.index + 1
    df_hout['timestamp'] = df_hout["time"].apply(lambda x: datetime.fromisoformat(x).timestamp())

    gafocal_df["timestamp"] = gafocal_df["time"].apply(lambda x: datetime.fromisoformat(x).timestamp())
    if use_gamma:
        event_info = gafocal_df[gafocal_df['comp_index'] == event_index]
        event_info = event_info.iloc[0]
    else:
        event_info = gafocal_df.loc[event_index]
    focal_dict = {
        'utc_time': event_info.time, 'timestamp': event_info.timestamp,
        'longitude': event_info.longitude, 'latitude': event_info.latitude,
        'depth': event_info.depth_km}
    focal_dict['focal_plane'] = {
        'strike': int(event_info.strike.split('+')[0]),
        'dip': int(event_info.dip.split('+')[0]),
        'rake': int(event_info.rake.split('+')[0])
        }
    focal_dict['quality_index'] = event_info.quality_index
    focal_dict['num_of_polarity'] = event_info.num_of_polarity
    df_test = df_hout[np.abs(df_hout["timestamp"] - event_info.timestamp) < 1]
    df_test = df_test[
        (np.abs(df_test["longitude"] - event_info.longitude) < 0.02) &  # Use tolerance for floats
        (np.abs(df_test["latitude"] - event_info.latitude) < 0.02)      # Use tolerance for floats
    ]

    tmp_index = df_test['tmp_index'].iloc[0]
    
    add_event_index(
        target_file=polarity_file,
        output_file=polarity_file_index,
        analyze_year=analyze_year
        )

    add_station_focal(
        polarity_index_file=polarity_file_index,
        focal_dict=focal_dict,
        analyze_year=analyze_year,
        event_index=tmp_index,
        sac_parent_dir=sac_parent_dir,
        sac_dir_name=sac_dir_name,
        source=source
    )
    return focal_dict      

def get_beach(source, focal_dict, color, ax):
    """Plot the beachball diagram."""
    mt = pmt.MomentTensor(
        strike=focal_dict['focal_plane']['strike'],
        dip=focal_dict['focal_plane']['dip'],
        rake=focal_dict['focal_plane']['rake']
        )
    ax.set_axis_off()
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.2, 1.2)
    projection = 'lambert'

    beachball.plot_beachball_mpl(
        mt, ax,
        position=(0., 0.),
        size=2.0,
        color_t=mpl_color(color),
        projection=projection,
        size_units='data')
    for sta in focal_dict['station_info'].keys():
        takeoff = focal_dict['station_info'][sta]['takeoff_angle']
        azi = focal_dict['station_info'][sta]['azimuth']
        polarity = focal_dict['station_info'][sta]['polarity']
        if polarity == ' ':
            continue
        # to spherical coordinates, r, theta, phi in radians
        # flip direction when takeoff is upward
        rtp = np.array([[
            1.0 if takeoff <= 90. else -1.,
            np.deg2rad(takeoff),
            np.deg2rad(90.-azi)]])
        # to 3D coordinates (x, y, z)
        points = beachball.numpy_rtp2xyz(rtp)

        # project to 2D with same projection as used in beachball
        x, y = beachball.project(points, projection=projection).T
        
        ax.plot(
            x, y,
            '+' if polarity == '+' else 'o',
            ms=10. if polarity == '+' else 10./np.sqrt(2.),
            mew=2.0,
            mec='black',
            mfc='none')
        ax.text(x+0.025, y+0.025, sta)
    ax.text(
        1.2, 
        -0.5, 
        f"{source}\nQuality index: {focal_dict['quality_index']}\nnum of station: {focal_dict['num_of_polarity']}\n\
Strike: {focal_dict['focal_plane']['strike']}\nDip: {focal_dict['focal_plane']['dip']}\n\
Rake: {focal_dict['focal_plane']['rake']}", 
        fontsize=20
        )
# plotter
def plot_polarity_waveform(focal_dict: dict, fig=None, gs=None, n_cols=4,
                           train_window=0.64, visual_window=2):
    if fig is None:
        fig = plt.figure(figsize=(36,20))
    focal_station_list = list(focal_dict['station_info'].keys())
    wavenum = len(focal_station_list)
    n_rows = math.ceil(wavenum / n_cols)
    gs = GridSpec(n_rows, n_cols, left=0, right=0.9, top=0.37, bottom=0, hspace=0.4, wspace=0.05)
    for index, station in enumerate(focal_station_list):
        polarity = focal_dict['station_info'][station]['polarity']
        ii = index // n_cols
        jj = index % n_cols
        ax = fig.add_subplot(gs[ii, jj])
        x_wide = focal_dict['station_info'][station]['visual_time']
        y_wide = focal_dict['station_info'][station]['visual_sac']
        ax.plot(x_wide, y_wide, color='k')
        ax.set_xlim(0, visual_window*2)
        ax.grid(True, alpha=0.7)
        x = focal_dict['station_info'][station]['train_time']
        y = focal_dict['station_info'][station]['train_sac']
        ax.plot(x,y, color='r')
        ax.set_title(f"{station}({polarity})", fontsize = 15)
        ax.set_xticklabels([]) # Remove the tick labels but keep the ticks
        ax.set_yticklabels([]) # Remove the tick labels but keep the ticks
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax.scatter(x[int(train_window*100)],y[int(train_window*100)], color = 'c', marker = 'o')
# plotter    
def plot_beachball_info(focal_dict_list: list[dict], name_list: list[str], fig=None, gs=None):
    if fig is None:
        fig = plt.figure(figsize=(36,20))
        gs = GridSpec(10, 18, left=0, right=0.9, top=0.95 , bottom=0.42 , wspace=0.1)
    if len(focal_dict_list) == 2:
        ax1 = fig.add_subplot(gs[:5, 5:9]) # focal axes-upper part
        ax2 = fig.add_subplot(gs[5:, 5:9])
        ax_list = [ax1, ax2]
        color_list = ['r', 'b']
    else:
        ax_list = [fig.add_subplot(gs[:5, 5:9])]
        color_list = ['r']
    for focal_dict, name, ax, color in zip(focal_dict_list, name_list, ax_list, color_list):
        get_beach(focal_dict=focal_dict, source=name, ax=ax, color=color)

def plot_seis(sac_dict: dict, df_phasenet_picks: pd.DataFrame,
               df_gamma_picks: pd.DataFrame, ax, bar_length=2):
    """
    plot the seismometer waveform for check.
    """
    for sta in list(sac_dict.keys()):
        if sta not in list(df_seis_gamma_picks['station_id']):
            ax.plot(
                sac_dict[sta]['time'], sac_dict[sta]['sac_data'], color='k',
                linewidth=0.4, alpha=0.25, zorder =1
                )
            ax.text(
                sac_dict[sta]['time'][-1]+1, sac_dict[sta]['distance'], sta,
                fontsize=4, verticalalignment='center', alpha=0.25
                )
        else:
            ax.plot(
                sac_dict[sta]['time'], sac_dict[sta]['sac_data'], color='k',
                linewidth=0.4, zorder =1
                )
            ax.text(
                sac_dict[sta]['time'][-1]+1, sac_dict[sta]['distance'], sta,
                fontsize=4, verticalalignment='center'
                )
    # all picks        
    ax.scatter(
        df_phasenet_picks[df_phasenet_picks['phase_type'] == 'P']['x'], 
        df_phasenet_picks[df_phasenet_picks['phase_type'] == 'P']['y'], 
        color='r', 
        s =1, 
        zorder =2
        )
    ax.scatter(
        df_phasenet_picks[df_phasenet_picks['phase_type'] == 'S']['x'], 
        df_phasenet_picks[df_phasenet_picks['phase_type'] == 'S']['y'], 
        color='c', 
        s =1, 
        zorder =2
        )     
    # aso picks
    ax.plot(
        [df_gamma_picks[df_gamma_picks['phase_type'] == 'P']['x'], df_gamma_picks[df_gamma_picks['phase_type'] == 'P']['x']], 
        [df_gamma_picks[df_gamma_picks['phase_type'] == 'P']['y'].to_numpy() - bar_length, df_gamma_picks[df_gamma_picks['phase_type'] == 'P']['y'].to_numpy() + bar_length], 
        color='r', 
        linewidth=0.7, 
        zorder =2
        )
    ax.plot(
        [df_gamma_picks[df_seis_gamma_picks['phase_type'] == 'S']['x'], df_gamma_picks[df_gamma_picks['phase_type'] == 'S']['x']], 
        [df_gamma_picks[df_seis_gamma_picks['phase_type'] == 'S']['y'].to_numpy() - bar_length, df_gamma_picks[df_gamma_picks['phase_type'] == 'S']['y'].to_numpy() + bar_length], 
        color='c', 
        linewidth=0.7, 
        zorder =2
        ) 
    ax.set_xlim(0,90)
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    ax.tick_params(axis='both', which='major', length=3, width=1, labelsize=5)
    #ax2.tick_params(axis='x', which='minor', length=4, width=0.5)
    ax.set_xlabel("Time (s)", fontsize = 7)

def plot_das(event_total_seconds: float, interval: int, ax, h5_parent_dir: Path,
             df_das_gamma_picks: pd.DataFrame,
             df_das_phasenet_picks: pd.DataFrame):
    
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
        df_das_phasenet_picks[df_das_phasenet_picks['phase_type'] == 'P']["channel_index"].values * dx, 
        df_das_phasenet_picks[df_das_phasenet_picks['phase_type'] == 'P']["phase_index"].values * dt, 
        c='r', 
        s=1, 
        alpha=0.05
        )
    ax.scatter(
        df_das_gamma_picks[df_das_gamma_picks['phase_type'] == 'P']["channel_index"].values * dx, 
        df_das_gamma_picks[df_das_gamma_picks['phase_type'] == 'P']["phase_index"].values * dt, 
        c='r', 
        s=1, 
        alpha=0.3
        )
    ax.scatter(
        df_das_phasenet_picks[df_das_phasenet_picks['phase_type'] == 'S']["channel_index"].values * dx, 
        df_das_phasenet_picks[df_das_phasenet_picks['phase_type'] == 'S']["phase_index"].values * dt, 
        c='c', 
        s=1, 
        alpha=0.05
        )
    ax.scatter(
        df_das_gamma_picks[df_das_gamma_picks['phase_type'] == 'S']["channel_index"].values * dx, 
        df_das_gamma_picks[df_das_gamma_picks['phase_type'] == 'S']["phase_index"].values * dt, 
        c='c', 
        s=1, 
        alpha=0.3
        )
    ax.scatter([], [], c="r", label="P")
    ax.scatter([], [], c="c", label="S")
    ax.legend()
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Time (s)")

def plot_waveform_check(sac_dict: dict,
                        df_seis_phasenet_picks=None, df_seis_gamma_picks=None,
                        df_das_phasenet_picks=None, df_das_gamma_picks=None,
                        event_total_seconds=None, h5_parent_dir=None,
                        fig=None, gs=None,
                        interval=300):

    if fig is None:
        fig = plt.figure(figsize=(36,20))
        gs = GridSpec(10, 18, left=0, right=0.9, top=0.95 , bottom=0.42 , wspace=0.1)
    ax = fig.add_subplot(gs[:, 9:18])
    # TODO: Design a suitable gs
    if df_seis_gamma_picks is not None and df_seis_phasenet_picks is not None:
        plot_seis(
            sac_dict=sac_dict,
            df_phasenet_picks=df_seis_phasenet_picks,
            df_gamma_picks=df_seis_gamma_picks,
            ax=ax
        )
    if df_das_gamma_picks is not None and df_das_phasenet_picks is not None and event_total_seconds is not None and h5_parent_dir is not None:
        plot_das(
            event_total_seconds=event_total_seconds, 
            interval=interval, 
            ax=ax, 
            h5_parent_dir=h5_parent_dir, 
            df_das_gamma_picks=df_das_gamma_picks, 
            df_das_phasenet_picks=df_das_phasenet_picks
        )
        


if __name__ == '__main__':
    figure_parent_dir = Path('/home/patrick/Work/playground/cwa_gamma/5s/fig')
    figure_parent_dir.mkdir(parents=True, exist_ok=True)
    # station
    seis_station_info = Path("/home/patrick/Work/EQNet/tests/hualien_0403/station_seis.csv")
    all_station_info = Path("/home/patrick/Work/EQNet/tests/hualien_0403/station_all.csv")
    das_station_info = Path("/home/patrick/Work/EQNet/tests/hualien_0403/station_das.csv")
    # catalog
    seis_catalog = Path("/home/patrick/Work/AutoQuake/GaMMA/results/Hualien_data/daily/20240403.csv")
    das_catalog = Path("/home/patrick/Work/EQNet/tests/hualien_0403/gamma_das/gamma_order.csv")
    combined_catalog = Path("/home/patrick/Work/EQNet/tests/hualien_0403/gamma_seis_das/gamma_order.csv")
    das_threshold_catalog = Path("/home/patrick/Work/EQNet/tests/hualien_0403/check_fig/das_threshold.csv")
    new_das = Path("/home/patrick/Work/EQNet/tests/hualien_0403/gamma_test/test_4/events_66_22.csv")
    # h3dd catalog
    combined_h3dd = Path("/home/patrick/Work/AutoQuake/Reloc2/seis_das_whole.dat_ch.hout")
    seis_h3dd = Path("/home/patrick/Work/EQNet/tests/hualien_0403/20240403.hout")

    # 2024/04/01-2024/04/17 mag catalog from CWA, comparing it to GaMMA
    sac_parent_dir = Path("/home/patrick/Work/AutoQuake_pamicoding/Hualien0403/20240401_20240428/Dataset")
    sac_dir_name = 'data_final'
    phasenet_picks_parent = Path('/home/patrick/Work/AutoQuake_pamicoding/Hualien0403/20240401_20240428/Results')
    gamma_catalog = Path("/home/patrick/Work/AutoQuake/test_format/gamma_events_20240401_20240417.csv")
    gamma_picks = Path("/home/patrick/Work/AutoQuake/test_format/gamma_picks_20240401_20240428.csv")
    dout_file = Path("/home/patrick/Work/AutoQuake/test_format/gamma_20240401_20240417.dout")
    hout_file = Path("/home/patrick/Work/AutoQuake/test_format/gamma_20240401_20240417.hout")
    hout_index_file = Path("/home/patrick/Work/AutoQuake/test_format/gamma_20240401_20240417.hout_index")
    output_file = Path("/home/patrick/Work/AutoQuake/test_format/gamma_20240401_20240417.dout_index")
    gamma_gafocal = Path("/home/patrick/Work/AutoQuake/test_format/gamma_gafocal_20240401_20240417_results.txt")
    cwa_gafocal = Path("/home/patrick/Work/AutoQuake/test_format/cwa_gafocal_20240401_20240417_results.txt")
    gamma_polarity = Path("/home/patrick/Work/AutoQuake/test_format/gamma_20240401_20240417_polarity.dout")
    gamma_polarity_index = Path("/home/patrick/Work/AutoQuake/test_format/gamma_20240401_20240417_polarity.dout_index")
    cwa_polarity = Path("/home/patrick/Work/AutoQuake/test_format/cwa_20240401_20240417_polarity.dout")
    cwa_polarity_index = Path("/home/patrick/Work/AutoQuake/test_format/cwa_20240401_20240417_polarity.dout_index")
    
    # 0403 DAS data
    h5_parent_dir = Path('/raid4/DAS_data/iDAS_MiDAS/hdf5/20240403_hdf5/')

    ## 0403 DAS + Seis in small region
    das_new_catalog = Path("/home/patrick/Work/EQNet/tests/hualien_0403/gamma_test/test_5/gamma_events.csv")
    das_new_picks = Path("/home/patrick/Work/EQNet/tests/hualien_0403/gamma_test/test_5/gamma_clean_picks.csv")
    das_new_station = Path("/home/patrick/Work/EQNet/tests/hualien_0403/new_das_seis_station.csv")
    polarity_picks = Path("/home/patrick/Work/EQNet/tests/hualien_0403/gamma_test/test_5/gamma_events_polarity.csv")
#%% ========= For comparing ==========    
    # packing the data, you can only put 1 in it.
    catalog_list = [gamma_gafocal, cwa_gafocal] # [the main catalog, another catalog you want to compare]
    name_list = ["GaMMA(gafocal)", "CWA(gafocal)"] # ["name of main catalog", "name of compared catalog"]
    catalog_dict = pack(catalog_list, name_list) # pack it as a dictionary

    gamma_common, cwa_common, gamma_only, cwa_only = catalog_compare(
        catalog=catalog_dict,
        tol=5
    )
#%% ========= For beachball ==========    
    gamma_focal_dict = find_compared_gafocal_polarity(
        gafocal_df=gamma_common, 
        hout_file=hout_file, 
        polarity_file=gamma_polarity,
        polarity_file_index=gamma_polarity_index,
        analyze_year='2024',
        event_index=1,
        use_gamma=True,
        source='GaMMA'
        )
    
    cwa_focal_dict = find_compared_gafocal_polarity(
        gafocal_df=cwa_common,  
        polarity_file=cwa_polarity,
        polarity_file_index=cwa_polarity_index,
        analyze_year='2024',
        event_index=1,
        use_gamma=False,
        source='CWA'
        )
#%% ========= For polarity waveform ==========            
    compared_event_index = find_index_backward(
        gamma_catalog=gamma_catalog,
        focal_dict=gamma_focal_dict
    )
#%% ========= For 順向的檢查圖 ==========        

    df_gamma_event, event_dict, df_gamma_picks = preprocess_gamma_csv(
        gamma_catalog=gamma_catalog,
        gamma_picks=gamma_picks,
        event_i=compared_event_index
        )
    
    df_all_phasenet_picks = process_phasenet_csv(
        phasenet_picks_parent=phasenet_picks_parent, 
        date=event_dict['date']
        )
    
    sac_dict = find_sac_data(
        event_time=event_dict['event_time'],
        date=event_dict['date'],
        event_point=event_dict['event_point'],
        station_list=seis_station_info,
        sac_parent_dir=sac_parent_dir,
        sac_dir_name=sac_dir_name,
        amplify_index=5,
    )
    df_seis_phasenet_picks, df_das_phasenet_picks = find_phasenet_pick(
        event_total_seconds=event_dict['event_total_seconds'],
        sac_dict=sac_dict, # for acquiring the dist between station and event.
        df_all_picks=df_all_phasenet_picks
    )
    df_seis_gamma_picks, df_das_gamma_picks = find_gamma_pick(
        df_gamma_picks=df_gamma_picks,
        sac_dict=sac_dict,
        event_total_seconds=event_dict['event_total_seconds']
    )


#%% ========= plotting ==========
    # plotting parameter
    seis_region = [119.7, 122.5, 21.7, 25.4]
    das_region = [121.62, 121.64, 24.02, 24.04]
    # plot the location
    check_loc_compare(
        station_list=seis_station_info,
        focal_dict_list=[gamma_focal_dict, cwa_focal_dict],
        name_list=['GaMMA', 'CWA'],
        seis_region=[119.7, 122.5, 21.7, 25.4]
        )
    # plot the beachball
    plot_beachball_info(
        focal_dict_list=[gamma_focal_dict, cwa_focal_dict],
        name_list=['GaMMA', 'CWA']
    )
    # plot the polarity waveform
    plot_polarity_waveform(
        focal_dict=gamma_focal_dict
    )
    # plot phasenet & association result
    plot_waveform_check(sac_dict=sac_dict) # plot seismometer data
