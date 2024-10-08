# import os
# import glob
# import math
import logging

# import calendar
# from datetime import datetime, timedelta
from pathlib import Path

import h5py

# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
# from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

# from collections import defaultdict
# from pyrocko import moment_tensor as pmt
# from pyrocko.plot import beachball, mpl_color
from _plot_base import convert_channel_index, get_total_seconds, station_mask

# from obspy.imaging.beachball import beach
from geopy.distance import geodesic

# import multiprocessing as mp
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from obspy import Trace, UTCDateTime, read


def preprocess_gamma_csv(
    gamma_catalog: Path, gamma_picks: Path, event_i: int
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
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

    """
    df_catalog = pd.read_csv(gamma_catalog)
    df_event = df_catalog[df_catalog['event_index'] == event_i].copy()
    df_event.loc[:, 'datetime'] = pd.to_datetime(df_event['time'])
    df_event.loc[:, 'ymd'] = df_event['datetime'].dt.strftime('%Y%m%d')
    df_event.loc[:, 'hour'] = df_event['datetime'].dt.hour
    df_event.loc[:, 'minute'] = df_event['datetime'].dt.minute
    df_event.loc[:, 'seconds'] = (
        df_event['datetime'].dt.second + df_event['datetime'].dt.microsecond / 1_000_000
    )
    event_dict = {
        'date': df_event['ymd'].iloc[0],
        'event_time': df_event['time'].iloc[0],
        'event_total_seconds': get_total_seconds(df_event['datetime'].iloc[0]),
        'event_lat': df_event['latitude'].iloc[0],
        'event_lon': df_event['longitude'].iloc[0],
        'event_point': (df_event['latitude'].iloc[0], df_event['longitude'].iloc[0]),
        'event_depth': df_event['depth_km'].iloc[0],
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
    df_all_picks['system'] = df_all_picks['station_id'].apply(
        lambda x: str(x).split('.')[0]
    )  # TW or MiDAS blablabla ***
    df_all_picks['station'] = df_all_picks['station_id'].apply(
        lambda x: str(x).split('.')[1]
    )  # LONT (station name) or A001 (channel name)
    return df_all_picks


def find_das_data(
    event_index: int,
    hdf5_parent_dir: Path,
    polarity_picks: Path,
    interval=300,
    train_window=0.64,
    visual_window=2,
    sampling_rate=100,
):
    """
    temp function, disentangle is needed.
    """
    df_pol = pd.read_csv(polarity_picks)
    df_pol.rename(columns={'station_id': 'station'}, inplace=True)
    df_pol_clean = df_pol[df_pol['event_index'] == event_index]
    df_pol_clean = df_pol_clean[station_mask(df_pol_clean)]
    das_plot_dict = {}
    for _, row in df_pol_clean.iterrows():
        if row.polarity == 'U':
            polarity = '+'
        elif row.polarity == 'D':
            polarity = '-'
        else:
            polarity = ' '
        total_seconds = get_total_seconds(pd.to_datetime(row.phase_time))
        index = int(total_seconds // interval)
        window = f'{interval*index}_{interval*(index+1)}.h5'
        try:
            file = list(hdf5_parent_dir.glob(f'*{window}'))[0]
        except IndexError:
            logging.info(f'File not found for window {window}')

        channel_index = convert_channel_index(row.station)
        tr = Trace()
        try:
            with h5py.File(file, 'r') as fp:
                ds = fp['data']
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
            tr.filter('bandpass', freqmin=1, freqmax=10)
            tr.taper(0.001)
            tr.resample(sampling_rate=sampling_rate)
            # this is for visualize length
            tr.trim(starttime=window_starttime_trim, endtime=window_endtime_trim)
            visual_time = np.arange(
                0, 2 * visual_window + 1 / sampling_rate, 1 / sampling_rate
            )  # using array to ensure the time length as same as time_window.
            visual_sac = tr.data
            # this is actual training length
            tr.trim(starttime=train_starttime_trim, endtime=train_endtime_trim)
            start_index = visual_window - train_window
            train_time = np.arange(
                start_index,
                start_index + 2 * train_window + 1 / sampling_rate,
                1 / sampling_rate,
            )  # using array to ensure the time length as same as time_window.
            train_sac = tr.data
            # final writing
            if 'station_info' not in das_plot_dict:
                das_plot_dict['station_info'] = {}
            das_plot_dict['station_info'][row.station] = {
                'polarity': polarity,
                'p_arrival': p_arrival,
                'visual_time': visual_time,
                'visual_sac': visual_sac,
                'train_time': train_time,
                'train_sac': train_sac,
            }
        except Exception as e:
            print(e)
    return das_plot_dict


def find_sac_data(
    event_time: str,
    date: str,
    event_point: tuple[float, float],
    station_list: Path,
    sac_parent_dir: Path,
    sac_dir_name: str,
    amplify_index: float,
    ins_type='seis',
) -> dict:
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
    # *** TODO: Adding the read hdf5 for DAS
    elif ins_type == 'DAS':
        df_station = df_station[df_station['station'].apply(lambda x: x[1].isdigit())]

    sac_path = sac_parent_dir / date / sac_dir_name
    sac_dict = {}
    for sta in df_station['station'].to_list():
        # glob the waveform
        try:
            data_path = list(sac_path.glob(f'*{sta}.*Z.*'))[0]
        except Exception:
            logging.info(f"we can't access the {sta}")
            continue

        sta_point = (
            df_station[df_station['station'] == sta]['latitude'].iloc[0],
            df_station[df_station['station'] == sta]['longitude'].iloc[0],
        )
        dist = geodesic(event_point, sta_point).km
        dist_round = np.round(dist, 1)

        # read the waveform
        st = read(data_path)
        st.taper(type='hann', max_percentage=0.05)
        st.filter('bandpass', freqmin=1, freqmax=10)
        st_check = True
        if starttime_trim < st[0].stats.starttime:
            st_check = False
        st[0].trim(starttime=starttime_trim, endtime=endtime_trim)
        sampling_rate = 1 / st[0].stats.sampling_rate
        time_sac = np.arange(
            0, 90 + sampling_rate, sampling_rate
        )  # using array to ensure the time length as same as time_window.
        x_len = len(time_sac)
        try:
            data_sac_raw = st[0].data / max(st[0].data)  # normalize the amplitude.
        except Exception as e:
            logging.info(f'Error: {e}')
            logging.info(f'check the length of given time: {len(st[0].data)}')
            continue
        data_sac_raw = data_sac_raw * amplify_index + dist
        # we might have the data lack in the beginning:
        if not st_check:
            data_sac = np.pad(
                data_sac_raw,
                (x_len - len(data_sac_raw), 0),
                mode='constant',
                constant_values=np.nan,
            )  # adding the Nan to ensure the data length as same as time window.
        else:
            data_sac = np.pad(
                data_sac_raw,
                (0, x_len - len(data_sac_raw)),
                mode='constant',
                constant_values=np.nan,
            )  # adding the Nan to ensure the data length as same as time window.

        sac_dict[str(sta)] = {
            'time': time_sac,
            'sac_data': data_sac,
            'distance': dist_round,
        }
    return sac_dict


def find_phasenet_pick(
    event_total_seconds: float,
    sac_dict: dict,
    df_all_picks: pd.DataFrame,
    first_half=30,
    second_half=60,
    dx=4.084,
    dt=0.01,
    hdf5_time=300,
):
    """
    Filtering waveform in specific time window and convert it for scatter plot.
    """
    time_window_start = event_total_seconds - first_half
    time_window_end = event_total_seconds + second_half
    pick_time = df_all_picks['total_seconds'].to_numpy()
    df_event_picks = df_all_picks[
        (pick_time >= time_window_start) & (pick_time <= time_window_end)
    ]

    df_seis_picks = df_event_picks[df_event_picks['station'].map(station_mask)]
    if not df_seis_picks.empty:
        df_seis_picks['x'] = (
            df_seis_picks['total_seconds'] - event_total_seconds + first_half
        )  # Because we use -30 as start.
        df_seis_picks['y'] = df_seis_picks['station'].map(
            lambda x: sac_dict[x]['distance']
        )

    df_das_picks = df_event_picks[~df_event_picks['station'].map(station_mask)]
    if not df_das_picks.empty:
        df_das_picks['channel_index'] = df_das_picks['station'].map(
            convert_channel_index
        )
        df_das_picks['x'] = (
            df_das_picks['station'].map(convert_channel_index) * dx
        )  # 4.084 = dx
        # TODO: This should thinks again about continuity.
        df_das_picks['y'] = df_das_picks['phase_index'].map(
            lambda x: x * dt + hdf5_time if x * dt - first_half < 0 else x * dt
        )  # 0.01 = dt

    return df_seis_picks, df_das_picks


def find_gamma_pick(
    df_gamma_picks: pd.DataFrame,
    sac_dict: dict,
    event_total_seconds: float,
    first_half=30,
    dx=4.084,
    dt=0.01,
    hdf5_time=300,
):
    """
    Preparing the associated picks for scatter plot.
    """
    df_das_aso_picks = df_gamma_picks[~df_gamma_picks['station_id'].apply(station_mask)]
    if not df_das_aso_picks.empty:
        df_das_aso_picks['channel_index'] = df_das_aso_picks['station_id'].apply(
            convert_channel_index
        )
        df_das_aso_picks['x'] = (
            df_das_aso_picks['station_id'].apply(convert_channel_index) * dx
        )
        df_das_aso_picks['y'] = df_das_aso_picks['phase_index'].apply(
            lambda x: x * dt + hdf5_time if x * dt - first_half < 0 else x * dt
        )

    df_seis_aso_picks = df_gamma_picks[df_gamma_picks['station_id'].apply(station_mask)]
    if not df_seis_aso_picks.empty:
        df_seis_aso_picks['total_seconds'] = df_seis_aso_picks['phase_time'].apply(
            get_total_seconds
        )
        df_seis_aso_picks['x'] = (
            df_seis_aso_picks['total_seconds'] - event_total_seconds + first_half
        )  # Because we use -30 as start.
        df_seis_aso_picks['y'] = df_seis_aso_picks['station_id'].map(
            lambda x: sac_dict[x]['distance']
        )

    return df_seis_aso_picks, df_das_aso_picks
