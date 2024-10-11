# import os
# import glob
# import math
import logging

# import calendar
# from datetime import datetime, timedelta
from pathlib import Path

import cartopy.crs as ccrs
import h5py
import matplotlib.pyplot as plt

# from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import pygmt

# from obspy.imaging.beachball import beach
from geopy.distance import geodesic
from matplotlib.colors import LightSource
from matplotlib.gridspec import GridSpec

# import multiprocessing as mp
# from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from obspy import Trace, UTCDateTime, read

# from collections import defaultdict
# from pyrocko import moment_tensor as pmt
# from pyrocko.plot import beachball, mpl_color
from ._plot_base import (
    check_format,
    convert_channel_index,
    get_total_seconds,
    plot_waveform_check,
    station_mask,
)


def preprocess_gamma_csv(
    gamma_catalog: Path, gamma_picks: Path, event_i: int, h3dd_hout=None
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
    h3dd_hout (optional, if you want to use h3dd info) : Path
        The path of h3dd.hout
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
    event_dict = {}
    event_dict[
        'gamma'
    ] = {}  # the reason I don't want to use defaultdict is to limit it.
    event_dict['gamma'] = {
        'date': df_event['ymd'].iloc[0],
        'event_time': df_event['time'].iloc[0],
        'event_total_seconds': get_total_seconds(df_event['datetime'].iloc[0]),
        'event_lat': df_event['latitude'].iloc[0],
        'event_lon': df_event['longitude'].iloc[0],
        'event_point': (df_event['latitude'].iloc[0], df_event['longitude'].iloc[0]),
        'event_depth': df_event['depth_km'].iloc[0],
    }
    if h3dd_hout is not None:
        event_dict['h3dd'] = {}
        df_h3dd, _ = check_format(h3dd_hout)
        df_h3dd['datetime'] = pd.to_datetime(df_h3dd['time'])
        df_h3dd['datetime'].map(get_total_seconds)
        # TODO: Using the first row is not always the correct way. We better have index to reference.
        target = df_h3dd[
            abs(
                df_h3dd['datetime'].map(get_total_seconds)
                - get_total_seconds(df_event['datetime'].iloc[0])
            )
            < 3
        ].iloc[0]
        event_dict['h3dd'] = {
            'event_time': target.time,
            'event_total_seconds': get_total_seconds(target.datetime),
            'event_lat': target.latitude,
            'event_lon': target.longitude,
            'event_point': (target.latitude, target.longitude),
            'event_depth': target.depth_km,
        }

    df_picks = pd.read_csv(gamma_picks)
    df_event_picks = df_picks[df_picks['event_index'] == event_i].copy()
    # TODO: does there have any possible scenario?
    # df_event_picks['station_id'] = df_event_picks['station_id'].map(lambda x: str(x).split('.')[1])
    df_event_picks.loc[:, 'phase_time'] = pd.to_datetime(df_event_picks['phase_time'])

    return df_event, event_dict, df_event_picks


def preprocess_phasenet_csv(
    phasenet_picks: Path, get_station=lambda x: str(x).split('.')[1]
):
    df_all_picks = pd.read_csv(phasenet_picks)
    df_all_picks['phase_time'] = pd.to_datetime(df_all_picks['phase_time'])
    df_all_picks['total_seconds'] = df_all_picks['phase_time'].apply(get_total_seconds)
    # df_all_picks['system'] = df_all_picks['station_id'].apply(
    #     lambda x: str(x).split('.')[0]
    # )  # TW or MiDAS blablabla
    df_all_picks['station'] = df_all_picks['station_id'].apply(
        get_station
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
    station_mask=station_mask,
):
    """
    temp function, disentangle is needed.
    """
    df_pol = pd.read_csv(polarity_picks)
    df_pol.rename(columns={'station_id': 'station'}, inplace=True)
    df_pol_clean = df_pol[df_pol['event_index'] == event_index]
    df_pol_clean = df_pol_clean[df_pol_clean['station'].map(station_mask)]
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
    station_mask=station_mask,
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
    station_mask=station_mask,
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


def get_mapview(region, ax):
    topo = (
        pygmt.datasets.load_earth_relief(resolution='15s', region=region).to_numpy()
        / 1e3
    )  # km
    x = np.linspace(region[0], region[1], topo.shape[1])
    y = np.linspace(region[2], region[3], topo.shape[0])
    dx, dy = 1, 1
    xgrid, ygrid = np.meshgrid(x, y)
    ls = LightSource(azdeg=0, altdeg=45)
    ax.pcolormesh(
        xgrid,
        ygrid,
        ls.hillshade(topo, vert_exag=10, dx=dx, dy=dy),
        vmin=-1,
        shading='gouraud',
        cmap='gray',
        alpha=1.0,
        antialiased=True,
        rasterized=True,
    )
    # cartopy setting
    ax.coastlines()
    ax.set_extent(region)


def plot_map(
    station: Path,
    event_dict: dict,
    df_event_picks: pd.DataFrame,
    seis_ax=None,
    das_ax=None,
    seis_region=None,
    das_region=None,
    use_gamma=True,
    use_h3dd=False,
    station_mask=station_mask,
):
    map_proj = ccrs.PlateCarree()
    # tick_proj = ccrs.PlateCarree()
    if seis_ax is None and das_ax is None:
        fig = plt.figure(figsize=(8, 12))
        gs = GridSpec(2, 1, height_ratios=[3, 1])
    df_station = pd.read_csv(station)
    df_seis_station = df_station[df_station['station'].map(station_mask)]
    df_das_station = df_station[~df_station['station'].map(station_mask)]
    if not df_seis_station.empty:
        if seis_ax is None:
            seis_ax = fig.add_subplot(gs[0], projection=map_proj)
        if seis_region is None:
            seis_region = [
                df_seis_station['longitude'].min() - 0.5,
                df_seis_station['longitude'].max() + 0.5,
                df_seis_station['latitude'].min() - 0.5,
                df_seis_station['latitude'].max() + 0.5,
            ]
        get_mapview(seis_region, seis_ax)
        seis_ax.scatter(
            x=df_seis_station['longitude'],
            y=df_seis_station['latitude'],
            marker='^',
            color='silver',
            edgecolors='k',
            s=50,
            zorder=2,
        )
        if use_gamma:
            seis_ax.scatter(
                x=event_dict['gamma']['event_lon'],
                y=event_dict['gamma']['event_lat'],
                marker='*',
                color='r',
                s=100,
                zorder=4,
                label='GaMMA',
            )
        if use_h3dd:
            seis_ax.scatter(
                x=event_dict['h3dd']['event_lon'],
                y=event_dict['h3dd']['event_lat'],
                marker='*',
                color='b',
                s=100,
                zorder=4,
                label='H3DD',
            )
        seis_ax.legend()
        for sta in df_event_picks[df_event_picks['station_id'].map(station_mask)][
            'station_id'
        ].unique():
            seis_ax.scatter(
                x=df_seis_station[df_seis_station['station'] == sta]['longitude'],
                y=df_seis_station[df_seis_station['station'] == sta]['latitude'],
                marker='^',
                color='darkorange',
                edgecolors='k',
                s=50,
                zorder=3,
            )
        seis_gl = seis_ax.gridlines(draw_labels=True)
        seis_gl.top_labels = False  # Turn off top labels
        seis_gl.right_labels = False
        seis_ax.autoscale(tight=True)
        seis_ax.set_aspect('auto')
    if not df_das_station.empty:
        if das_ax is None:
            das_ax = fig.add_subplot(gs[1], projection=map_proj)
        if das_region is None:
            das_region = [
                df_das_station['longitude'].min() - 0.01,
                df_das_station['longitude'].max() + 0.01,
                df_das_station['latitude'].min() - 0.01,
                df_das_station['latitude'].max() + 0.01,
            ]
        get_mapview(das_region, das_ax)
        das_ax.scatter(
            x=df_das_station['longitude'],
            y=df_das_station['latitude'],
            marker='.',
            color='silver',
            s=5,
            zorder=2,
        )
        if use_gamma:
            das_ax.scatter(
                x=event_dict['gamma']['event_lon'],
                y=event_dict['gamma']['event_lat'],
                marker='*',
                color='r',
                s=100,
                zorder=4,
                label='GaMMA',
            )
        if use_h3dd:
            das_ax.scatter(
                x=event_dict['h3dd']['event_lon'],
                y=event_dict['h3dd']['event_lat'],
                marker='*',
                color='b',
                s=100,
                zorder=4,
                label='H3DD',
            )
        das_ax.legend()
        for sta in df_event_picks[~df_event_picks['station_id'].map(station_mask)][
            'station_id'
        ].unique():
            das_ax.scatter(
                x=df_das_station[df_das_station['station'] == sta]['longitude'],
                y=df_das_station[df_das_station['station'] == sta]['latitude'],
                marker='.',
                color='darkorange',
                s=5,
                zorder=3,
            )
        das_gl = das_ax.gridlines(draw_labels=True)
        das_gl.top_labels = False  # Turn off top labels
        das_gl.right_labels = False
        # das_ax.autoscale(tight=True)
        # das_ax.set_aspect('auto')


def return_none_if_empty(df):
    if df.empty:
        return None
    return df


def plot_asso(
    df_phasenet_picks: pd.DataFrame,
    gamma_picks: Path,
    gamma_events: Path,
    station: Path,
    event_i: int,
    fig_dir: Path,
    h3dd_hout=None,
    amplify_index=5,
    sac_parent_dir=None,
    sac_dir_name=None,
    h5_parent_dir=None,
    station_mask=station_mask,
    seis_region=None,
    das_region=None,
):
    """
    ## Plotting the gamma and h3dd info.

    Args:
        - df_phasenet_picks (pd.DataFrame): phasenet picks.
        - gamma_picks (Path): Path to the gamma picks.
        - gamma_events (Path): Path to the gamma events.
        - station (Path): Path to the station info.
        - event_i (int): Index of the event to plot.
        - fig_dir (Path): Path to save the figure.

        - h3dd_hout (Path, optional): Path to the h3dd hout. Defaults to None.
        - amplify_index (int, optional): Amplify index for sac data. Defaults to 5.
        - sac_parent_dir (Path, optional): Path to the sac data. Defaults to None.
        - sac_dir_name (Path, optional): Name of the sac data directory. Defaults to None.
        - h5_parent_dir (Path, optional): Path to the h5 data (DAS). Defaults to None.
        - station_mask (function, optional): Function to mask the station. Defaults to station_mask.
        - seis_region (list, optional): Region for seismic plot. Defaults to None.
            - 4-element list: [min_lon, max_lon, min_lat, max_lat]
        - das_region (list, optional): Region for DAS plot. Defaults to None.
            - 4-element list: [min_lon, max_lon, min_lat, max_lat]
    """
    df_event, event_dict, df_event_picks = preprocess_gamma_csv(
        gamma_catalog=gamma_events,
        gamma_picks=gamma_picks,
        event_i=event_i,
        h3dd_hout=h3dd_hout,
    )

    if h3dd_hout is not None:
        status = 'h3dd'
        use_h3dd = True
    else:
        status = 'gamma'
        use_h3dd = False

    # figure setting
    fig = plt.figure()
    map_proj = ccrs.PlateCarree()
    # tick_proj = ccrs.PlateCarree()

    # retrieving data
    if sac_parent_dir is not None and sac_dir_name is not None:
        sac_dict = find_sac_data(
            event_time=event_dict[status]['event_time'],
            date=event_dict['gamma']['date'],
            event_point=event_dict[status]['event_point'],
            station_list=station,
            sac_parent_dir=sac_parent_dir,
            sac_dir_name=sac_dir_name,
            amplify_index=amplify_index,
        )
        seis_map_ax = fig.add_axes([0.3, 0.5, 0.4, 0.8], projection=map_proj)
        seis_map_ax.set_title(
            f"Event_{event_i}: {event_dict[status]['event_time']}\nlat: {event_dict[status]['event_lat']}, lon: {event_dict[status]['event_lon']}, depth: {event_dict[status]['event_depth']} km"
        )
        seis_waveform_ax = fig.add_axes([0.82, 0.5, 0.8, 0.9])
    else:
        seis_map_ax = None
        seis_waveform_ax = None
        sac_dict = {}

    if h5_parent_dir is not None:
        das_map_ax = fig.add_axes([0.3, -0.15, 0.4, 0.55], projection=map_proj)
        das_waveform_ax = fig.add_axes([0.82, -0.15, 0.8, 0.55])
    else:
        das_map_ax = None
        das_waveform_ax = None
    df_seis_phasenet_picks, df_das_phasenet_picks = find_phasenet_pick(
        event_total_seconds=event_dict[status]['event_total_seconds'],
        sac_dict=sac_dict,
        df_all_picks=df_phasenet_picks,
        station_mask=station_mask,
    )
    df_seis_gamma_picks, df_das_gamma_picks = find_gamma_pick(
        df_gamma_picks=df_event_picks,
        sac_dict=sac_dict,
        event_total_seconds=event_dict[status]['event_total_seconds'],
        station_mask=station_mask,
    )
    plot_waveform_check(
        sac_dict=sac_dict,
        df_seis_phasenet_picks=return_none_if_empty(df_seis_phasenet_picks),
        df_seis_gamma_picks=return_none_if_empty(df_seis_gamma_picks),
        df_das_phasenet_picks=return_none_if_empty(df_das_phasenet_picks),
        df_das_gamma_picks=return_none_if_empty(df_das_gamma_picks),
        event_total_seconds=event_dict[status]['event_total_seconds'],
        h5_parent_dir=h5_parent_dir,
        das_ax=das_waveform_ax,
        seis_ax=seis_waveform_ax,
    )

    plot_map(
        station=station,
        event_dict=event_dict,
        df_event_picks=df_event_picks,
        seis_ax=seis_map_ax,
        das_ax=das_map_ax,
        use_h3dd=use_h3dd,
        station_mask=station_mask,
        seis_region=seis_region,
        das_region=das_region,
    )
    save_name = (
        event_dict[status]['event_time']
        .replace(':', '_')
        .replace('-', '_')
        .replace('.', '_')
    )

    plt.tight_layout()
    plt.savefig(
        fig_dir / f'event_{event_i}_{save_name}.png', bbox_inches='tight', dpi=300
    )
    plt.close()
