import calendar
import logging
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygmt
from cartopy.mpl.geoaxes import GeoAxes
from geopy.distance import geodesic
from matplotlib.colors import LightSource
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
from obspy import Stream, Trace, UTCDateTime, read
from pyrocko import moment_tensor as pmt
from pyrocko.plot import beachball, mpl_color


# ===== Sauce =====
def polarity_color_select(polarity: str):
    if polarity == 'x':
        return 'k'
    elif polarity == 'U':
        return 'r'
    elif polarity == 'D':
        return 'b'


def normalize(x):
    """## For plotting DAS"""
    return (x - np.mean(x, axis=-1, keepdims=True)) / np.std(x, axis=-1, keepdims=True)


def formatting(num: str):  # *** why we don't use the :02 in formatting
    """Format single digit numbers with a leading zero."""
    return f'0{num}' if len(num) == 1 else num


def get_total_seconds(dt: pd.Timestamp | str) -> float:
    """## Convert a datetime string into total seconds."""
    if isinstance(dt, str):
        dt = pd.to_datetime(dt)
    return (dt - dt.normalize()).total_seconds()


def utc_to_timestamp(utc_string: str):
    """Converts an ISO format string to a Unix timestamp."""
    return datetime.fromisoformat(utc_string).timestamp()


def quick_line_check(path):
    with open(path) as r:
        lines = r.readlines()
    counter = 0
    for i in lines:
        if i.strip()[:4] == '2024':
            counter += 1
    return counter


def h3dd_time_trans(x: str):
    """Converting the h3dd time format into total seconds in a day."""
    return int(x[:2]) * 3600 + int(x[2:4]) * 60 + int(x[4:6]) + float(f'0.{x[6:]}')


def convert_channel_index(sta_name: str) -> int:
    """Convert first character of channel from alpha to digit for indexing."""
    if sta_name[:1] == 'A':
        channel_index = int(sta_name[1:])
    elif sta_name[:1] == 'B':
        channel_index = int(f'1{sta_name[1:]}')
    else:
        print('wrong format warning: please append the condition')
    return channel_index


def station_mask(x: str):
    """Pattern to distinguish the DAS and Seismometer station."""
    return x[1].isalpha()


def degree_trans(part: str):
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
    return deg + dig


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
        sec = sec % 60  # Keep remaining seconds

    # Handle minute overflow
    if min >= 60:
        hour += min // 60  # Increment hours by minute overflow
        min = min % 60  # Keep remaining minutes

    # Handle hour overflow
    if hour >= 24:
        day += hour // 24  # Increment days by hour overflow
        hour = hour % 24  # Keep remaining hours

    # Handle day overflow (check if day exceeds days in the current month)
    while day > calendar.monthrange(year, month)[1]:  # Get number of days in month
        day -= calendar.monthrange(year, month)[1]  # Subtract days in current month
        month += 1  # Increment month

        # Handle month overflow
        if month > 12:
            month = 1
            year += 1  # Increment year if month overflows

    return year, month, day, hour, min, sec


def check_hms_h3dd(hms: str):
    """check whether the h3dd format's second overflow"""
    minute = int(hms[2:4])
    second = int(hms[4:6])

    if second >= 60:
        minute += second // 60
        second = second % 60

    fixed_hms = hms[:2] + f'{minute:02d}' + f'{second:02d}' + hms[6:]
    return fixed_hms


def _cal_dist(event_point: tuple, sta_point: tuple):
    """## round distance between station and event."""
    dist = geodesic(event_point, sta_point).km
    dist_round = np.round(dist, 1)
    return dist_round


def check_hms_gafocal(hms: str):
    """
    check whether the gafocal format's second overflow
    """
    minute = int(hms[3:5])
    second = int(hms[6:8])

    if second >= 60:
        minute += second // 60
        second = second % 60

    fixed_hms = hms[:3] + f'{minute:02d}' + hms[5:6] + f'{second:02d}'
    return fixed_hms


def _txt_preprocessor(df: pd.DataFrame) -> pd.DataFrame:
    """
    Distinguish h3dd and gafocal format through YYYYMMDD & YYYY/MM/DD.
    """
    if len(df[0].iloc[0]) == 8:  # h3dd
        df[1] = df[1].apply(check_hms_h3dd)
        df['time'] = (
            df[0].apply(lambda x: f'{x[:4]}-{x[4:6]}-{x[6:8]}')
            + 'T'
            + df[1].apply(lambda x: f'{x[:2]}:{x[2:4]}:{x[4:6]}.{x[6:8]}0000')
        )
        df = df.rename(columns={2: 'latitude', 3: 'longitude', 4: 'depth_km'})
        mask = [i for i in df.columns.tolist() if isinstance(i, str)]
        df = df[mask]
        return df
    elif '/' in df[0].iloc[0]:  # gafocal
        df[1] = df[1].apply(check_hms_gafocal)
        df['time'] = df[0].apply(lambda x: x.replace('/', '-')) + 'T' + df[1]
        df = df.rename(
            columns={
                2: 'longitude',
                3: 'latitude',
                4: 'depth_km',
                5: 'magnitude',
                6: 'strike',
                7: 'strike_err',
                8: 'dip',
                9: 'dip_err',
                10: 'rake',
                11: 'rake_err',
                12: 'quality_index',
                13: 'num_of_polarity',
            }
        )
        mask = [i for i in df.columns.tolist() if isinstance(i, str)]
        df = df[mask]
        return df
    else:
        raise ValueError(f'Unrecognized date format: {df[0].iloc[0]}')


def _merge_latest(data_path: Path, sta_name: str):
    """
    Merge the latest data from the given data path for the specified station name.
    """
    sac_list = list(data_path.glob(f'*{sta_name}*Z*'))
    if not sac_list:
        logging.info(f'{sta_name} using other component')
        sac_list = list(data_path.glob(f'*{sta_name}*'))
        logging.info(f'comp_list: {sac_list}')
    stream = Stream()
    for sac_file in sac_list:
        st = read(sac_file)
        stream += st
    stream = stream.merge(fill_value='latest')
    return stream


def get_max_columns(file_path):
    max_columns = 0

    # Open the file and analyze each line
    with open(file_path) as file:
        for line in file:
            # Split the line using the space delimiter and count the columns
            columns_in_line = len(line.split())
            # Update max_columns if this line has more columns
            if columns_in_line > max_columns:
                max_columns = columns_in_line

    return max_columns


def check_format(catalog, i=None):
    """
    Checking the format is GaMMA (events_catalog.csv), h3dd (.hout), gafocal (.txt), and
    converting it into union format.
    """
    if isinstance(catalog, dict):
        file_path = catalog[i]['catalog']
    elif isinstance(catalog, Path):
        file_path = catalog
    else:
        raise ValueError('Invalid input. Provide either a dictionary or a Path object.')

    if file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
    else:
        max_columns = get_max_columns(file_path)

        cols_to_read = list(range(max_columns - 1))
        df = pd.read_csv(
            file_path,
            sep=r'\s+',
            header=None,
            dtype={0: 'str', 1: 'str'},
            usecols=cols_to_read,
        )
        df = _txt_preprocessor(df)
    timestamp = df['time'].apply(utc_to_timestamp).to_numpy()
    return df, timestamp


# ===== preprocess function =====


def preprocess_gamma_csv(
    gamma_catalog: Path, gamma_picks: Path, event_i: int, h3dd_hout=None
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    ## Preprocessing the DataFrame.

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


def _preprocess_phasenet_csv(
    phasenet_picks: Path, get_station=lambda x: str(x).split('.')[1]
):
    """## Preprocess the phasenet_csv"""
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


def _process_polarity(
    polarity_picks: Path, event_index: int, station_mask=station_mask
):
    """
    ### TODO:Actually station mask should not exist here, this place should already contains
    the data of DAS.
    """
    df_pol = pd.read_csv(polarity_picks)
    df_pol.rename(columns={'station_id': 'station'}, inplace=True)
    df_pol = df_pol[df_pol['event_index'] == event_index]
    df_pol = df_pol[df_pol['station'].map(station_mask)]
    return df_pol


def _midas_scenario(
    phase_time: str, hdf5_parent_dir: Path, interval=300
) -> None | Path:
    total_seconds = get_total_seconds(pd.to_datetime(phase_time))
    index = int(total_seconds // interval)
    window = f'{interval*index}_{interval*(index+1)}.h5'
    try:
        file = list(hdf5_parent_dir.glob(f'*{window}'))[0]
        return file
    except IndexError:
        logging.info(f'File not found for window {window}')
        return None


def _process_das_polarity(
    file: Path,
    channel_index: int,
    phase_time: str,
    train_window=0.64,
    visual_window=2.0,
    sampling_rate=100,
) -> tuple:
    tr = Trace()
    try:
        with h5py.File(file, 'r') as fp:
            ds = fp['data']
            data = ds[channel_index]
            tr.stats.sampling_rate = 1 / ds.attrs['dt_s']
            tr.stats.starttime = ds.attrs['begin_time']
            tr.data = data
        p_arrival = UTCDateTime(phase_time)
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
        return p_arrival, visual_time, visual_sac, train_time, train_sac
    except Exception as e:
        logging.info(f'Error {e} during hdf5 and obspy process')
        return ()


# ===== Find function =====


def find_gamma_h3dd(gamma_events: Path, h3dd_events: Path, event_index: int):
    """Using gamma index to find the h3dd event."""
    df_gamma_events = pd.read_csv(gamma_events)
    h3dd_event_index = df_gamma_events[df_gamma_events['event_index'] == event_index][
        'h3dd_event_index'
    ].iloc[0]

    df_h3dd, _ = check_format(catalog=h3dd_events)
    df_h3dd.loc[h3dd_event_index]
    return df_h3dd


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
    """## find the das waveform for plotting polarity waveform.
    ### TODO: We use MiDAS Naming idiom to design the searching way, it should be flexible with
    other format, but that's what we need to improve!
    """
    df_pol = _process_polarity(polarity_picks, event_index, station_mask)
    das_plot_dict = {}
    for _, row in df_pol.iterrows():
        # Polarity determination
        if row.polarity == 'U':
            polarity = '+'
        elif row.polarity == 'D':
            polarity = '-'
        else:
            polarity = ' '

        # file scenario
        file = _midas_scenario(row.phase_time, hdf5_parent_dir, interval)
        if file is None:
            continue

        # Convert A001 back to 1001 for hdf5 key.
        channel_index = convert_channel_index(row.station)
        try:
            p_arrival, visual_time, visual_sac, train_time, train_sac = (
                _process_das_polarity(
                    file=file,
                    channel_index=channel_index,
                    phase_time=row.phase_time,
                    train_window=train_window,
                    visual_window=visual_window,
                    sampling_rate=sampling_rate,
                )
            )
        except Exception as e:
            logging.info(f'Error: {e}, because it returns a tuple')
            continue

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
    return das_plot_dict


def find_sac_data(
    event_time: str,
    date: str,  # redundant, but already do it in previous function, temp pass.
    event_point: tuple[float, float],
    station: Path,
    sac_parent_dir: Path,
    amplify_index: float,
    station_mask=station_mask,
) -> dict:
    """
    ## Retrieve the waveform from SAC.

    This function glob the sac file from each station, multiplying the
    amplify index to make the PGA/PGV more clear, then packs the
    waveform data into dictionary for plotting.

    This is for plotting association plot!
    """
    starttime_trim = UTCDateTime(event_time) - 30
    endtime_trim = UTCDateTime(event_time) + 60

    df_station = pd.read_csv(station)
    df_station = df_station[df_station['station'].map(station_mask)]

    sac_path = sac_parent_dir / date
    sac_dict = {}
    for sta in df_station['station'].to_list():
        # glob the waveform

        st = _merge_latest(data_path=sac_path, sta_name=sta)
        st.taper(type='hann', max_percentage=0.05)
        st.filter('bandpass', freqmin=1, freqmax=10)
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

        sta_point = (
            df_station[df_station['station'] == sta]['latitude'].iloc[0],
            df_station[df_station['station'] == sta]['longitude'].iloc[0],
        )
        dist = _cal_dist(event_point=event_point, sta_point=sta_point)
        data_sac_raw = data_sac_raw * amplify_index + dist
        # we might have the data lack in the beginning:
        if starttime_trim < st[0].stats.starttime:
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
            'distance': dist,
        }
    return sac_dict


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


def find_phasenet_pick(
    event_total_seconds: float,
    sac_dict: dict,
    df_all_picks: pd.DataFrame,  # why this place using dataframe? because we don't want to read it several time.
    first_half=30,
    second_half=60,
    dx=4.084,
    dt=0.01,
    das_time_interval=300,
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
        df_das_picks['x'] = df_das_picks['station'].map(convert_channel_index) * dx
        # TODO: This should thinks again about continuity.
        df_das_picks['y'] = df_das_picks['phase_index'].map(
            lambda x: x * dt + das_time_interval if x * dt - first_half < 0 else x * dt
        )  # 0.01 = dt

    return df_seis_picks, df_das_picks


# ===== Plot private function =====
def _plot_seis(
    sac_dict: dict,
    df_phasenet_picks: pd.DataFrame,
    df_gamma_picks: pd.DataFrame,
    ax,
    bar_length=2,
):
    """
    plot the seismometer waveform for check.
    """
    for sta in list(sac_dict.keys()):
        if sta not in list(df_gamma_picks['station_id']):
            ax.plot(
                sac_dict[sta]['time'],
                sac_dict[sta]['sac_data'],
                color='k',
                linewidth=0.4,
                alpha=0.25,
                zorder=1,
            )
            ax.text(
                sac_dict[sta]['time'][-1] + 1,
                sac_dict[sta]['distance'],
                sta,
                fontsize=4,
                verticalalignment='center',
                alpha=0.25,
            )
        else:
            ax.plot(
                sac_dict[sta]['time'],
                sac_dict[sta]['sac_data'],
                color='k',
                linewidth=0.4,
                zorder=1,
            )
            ax.text(
                sac_dict[sta]['time'][-1] + 1,
                sac_dict[sta]['distance'],
                sta,
                fontsize=4,
                verticalalignment='center',
            )
    # all picks
    ax.scatter(
        df_phasenet_picks[df_phasenet_picks['phase_type'] == 'P']['x'],
        df_phasenet_picks[df_phasenet_picks['phase_type'] == 'P']['y'],
        color='r',
        s=1,
        zorder=2,
    )
    ax.scatter(
        df_phasenet_picks[df_phasenet_picks['phase_type'] == 'S']['x'],
        df_phasenet_picks[df_phasenet_picks['phase_type'] == 'S']['y'],
        color='c',
        s=1,
        zorder=2,
    )
    # aso picks
    ax.plot(
        [
            df_gamma_picks[df_gamma_picks['phase_type'] == 'P']['x'],
            df_gamma_picks[df_gamma_picks['phase_type'] == 'P']['x'],
        ],
        [
            df_gamma_picks[df_gamma_picks['phase_type'] == 'P']['y'].to_numpy()
            - bar_length,
            df_gamma_picks[df_gamma_picks['phase_type'] == 'P']['y'].to_numpy()
            + bar_length,
        ],
        color='r',
        linewidth=0.7,
        zorder=2,
    )
    ax.plot(
        [
            df_gamma_picks[df_gamma_picks['phase_type'] == 'S']['x'],
            df_gamma_picks[df_gamma_picks['phase_type'] == 'S']['x'],
        ],
        [
            df_gamma_picks[df_gamma_picks['phase_type'] == 'S']['y'].to_numpy()
            - bar_length,
            df_gamma_picks[df_gamma_picks['phase_type'] == 'S']['y'].to_numpy()
            + bar_length,
        ],
        color='c',
        linewidth=0.7,
        zorder=2,
    )
    ax.set_xlim(0, 90)
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    ax.tick_params(axis='both', which='major', length=3, width=1, labelsize=5)
    # ax2.tick_params(axis='x', which='minor', length=4, width=0.5)
    ax.set_xlabel('Time (s)', fontsize=7)


# ===== add function =====


# ===== get function =====
def get_mapview(region: list, ax: GeoAxes, title='Map'):
    """## Plotting the basic map.

    Notice here, the ax should be a GeoAxes! A subclass of `matplotlib.axes.Axes`.
    """
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
    ax.set_title(title, fontsize=20)
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False  # Turn off top labels
    gl.right_labels = False


# ===== run function =====
# ===== plot function =====
def _plot_das(
    event_total_seconds: float,
    interval: int,
    ax,
    h5_parent_dir: Path,
    df_das_gamma_picks: pd.DataFrame,
    df_das_phasenet_picks: pd.DataFrame,
):
    index = int(event_total_seconds // interval)
    window = [f'{interval*index}_{interval*(index+1)}.h5']
    time_index = round(event_total_seconds % interval, 3)
    get_previous = False
    if time_index - 30 < 0 and index != 0:
        previous_window = f'{interval*(index-1)}_{interval*index}.h5'
        window.insert(0, previous_window)
        get_previous = True
    if time_index + 60 > interval and index != 287:
        next_window = f'{interval*(index+1)}_{interval*(index+2)}.h5'
        window.append(next_window)

    try:
        all_data = []
        for win in window:
            file = list(h5_parent_dir.glob(f'*{win}'))[0]
            if not file:
                raise IndexError(f'File not found for window {win}')
            try:
                with h5py.File(file, 'r') as fp:
                    ds = fp['data']
                    data = ds[...]  # np.array
                    dt = ds.attrs['dt_s']  # 0.01 sampling rate
                    dx = ds.attrs['dx_m']  # interval of cable ~ 4
                    nx, nt = data.shape
                    logging.info(data.shape)
                    x = np.arange(nx) * dx
                    t = np.arange(nt) * dt
                    all_data.append(data)
            except Exception as e:
                logging.info(f'Error reading {file}: {e}')
    except IndexError:
        logging.info(f'File not found for window {window}')

    # Handle the case where there is only one array in all_data
    if len(all_data) == 1:
        concatenated_data = all_data[0]
        logging.info('Only one data array, no need to concatenate.')
    elif len(all_data) > 1:
        # Concatenate all data arrays along the second axis (horizontally)
        concatenated_data = np.concatenate(all_data, axis=1)
        logging.info(f'Concatenated data shape: {concatenated_data.shape}')
    nx, nt = concatenated_data.shape
    x = np.arange(nx) * dx
    t = np.arange(nt) * dt

    ax.imshow(
        normalize(concatenated_data).T,
        cmap='seismic',
        vmin=-1,
        vmax=1,
        aspect='auto',
        extent=[x[0], x[-1], t[-1], t[0]],
        interpolation='none',
    )
    if get_previous:
        ax.set_ylim(
            time_index + 60 + interval, time_index - 30 + interval
        )  # because the concat, origin time should add interval.
    else:
        ax.set_ylim(
            time_index + 60, time_index - 30
        )  # concat later or not would not influence the time.
    ax.scatter(
        df_das_phasenet_picks[df_das_phasenet_picks['phase_type'] == 'P'][
            'channel_index'
        ].values
        * dx,
        df_das_phasenet_picks[df_das_phasenet_picks['phase_type'] == 'P'][
            'phase_index'
        ].values
        * dt,
        c='r',
        s=1,
        alpha=0.05,
    )
    ax.scatter(
        df_das_gamma_picks[df_das_gamma_picks['phase_type'] == 'P'][
            'channel_index'
        ].values
        * dx,
        df_das_gamma_picks[df_das_gamma_picks['phase_type'] == 'P'][
            'phase_index'
        ].values
        * dt,
        c='r',
        s=1,
        alpha=0.3,
    )
    ax.scatter(
        df_das_phasenet_picks[df_das_phasenet_picks['phase_type'] == 'S'][
            'channel_index'
        ].values
        * dx,
        df_das_phasenet_picks[df_das_phasenet_picks['phase_type'] == 'S'][
            'phase_index'
        ].values
        * dt,
        c='c',
        s=1,
        alpha=0.05,
    )
    ax.scatter(
        df_das_gamma_picks[df_das_gamma_picks['phase_type'] == 'S'][
            'channel_index'
        ].values
        * dx,
        df_das_gamma_picks[df_das_gamma_picks['phase_type'] == 'S'][
            'phase_index'
        ].values
        * dt,
        c='c',
        s=1,
        alpha=0.3,
    )
    ax.scatter([], [], c='r', label='P')
    ax.scatter([], [], c='c', label='S')
    ax.legend()
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Time (s)')


def plot_waveform_check(
    sac_dict: dict,
    df_seis_phasenet_picks: pd.DataFrame | None = None,
    df_seis_gamma_picks: pd.DataFrame | None = None,
    df_das_phasenet_picks: pd.DataFrame | None = None,
    df_das_gamma_picks: pd.DataFrame | None = None,
    event_total_seconds=None,
    h5_parent_dir=None,
    das_ax=None,
    seis_ax=None,
    interval=300,
):
    if das_ax is None and seis_ax is None:
        fig = plt.figure(figsize=(8, 12))
        gs = GridSpec(2, 1, height_ratios=[3, 1])
        seis_ax = fig.add_subplot(gs[0])
        das_ax = fig.add_subplot(gs[1])
    # TODO: Design a suitable gs
    if df_seis_gamma_picks is not None and df_seis_phasenet_picks is not None:
        _plot_seis(
            sac_dict=sac_dict,
            df_phasenet_picks=df_seis_phasenet_picks,
            df_gamma_picks=df_seis_gamma_picks,
            ax=seis_ax,
        )
    if (
        df_das_gamma_picks is not None
        and df_das_phasenet_picks is not None
        and event_total_seconds is not None
        and h5_parent_dir is not None
    ):
        _plot_das(
            event_total_seconds=event_total_seconds,
            interval=interval,
            ax=das_ax,
            h5_parent_dir=h5_parent_dir,
            df_das_gamma_picks=df_das_gamma_picks,
            df_das_phasenet_picks=df_das_phasenet_picks,
        )


def plot_station(df_station: pd.DataFrame, geo_ax):
    """
    plot the station distribution on map.
    """
    mask_alpha = df_station['station'].map(station_mask)
    mask_digit = ~mask_alpha
    if mask_digit.any():
        geo_ax.scatter(
            df_station[mask_digit]['longitude'],
            df_station[mask_digit]['latitude'],
            s=5,
            c='k',
            marker='.',
            alpha=0.5,
            rasterized=True,
            label='DAS',
        )
    if mask_alpha.any():
        geo_ax.scatter(
            df_station[mask_alpha]['longitude'],
            df_station[mask_alpha]['latitude'],
            s=50,
            c='c',
            marker='^',
            alpha=0.7,
            rasterized=True,
            label='Seismometer',
        )


def get_das_beach(
    focal_dict,
    ax,
    xlim=(-1.1, 1.1),
    ylim=(-1.1, 1.1),
    color='silver',
    only_das=False,
    add_info=False,
    source='AutoQuake',
):
    """Plot the beachball diagram with DAS."""
    mt = pmt.MomentTensor(
        strike=focal_dict['focal_plane']['strike'],
        dip=focal_dict['focal_plane']['dip'],
        rake=focal_dict['focal_plane']['rake'],
    )

    projection = 'lambert'

    beachball.plot_beachball_mpl(
        mt,
        ax,
        position=(0.0, 0.0),
        size=2.0,
        color_t=mpl_color(color),
        projection=projection,
        size_units='data',
    )
    for sta in focal_dict['station_info'].keys():
        takeoff = focal_dict['station_info'][sta]['takeoff_angle']
        azi = focal_dict['station_info'][sta]['azimuth']
        polarity = focal_dict['station_info'][sta]['polarity']
        # if polarity == ' ':
        #     continue
        # to spherical coordinates, r, theta, phi in radians
        # flip direction when takeoff is upward
        rtp = np.array(
            [
                [
                    1.0 if takeoff <= 90.0 else -1.0,
                    np.deg2rad(takeoff),
                    np.deg2rad(90.0 - azi),
                ]
            ]
        )
        # to 3D coordinates (x, y, z)
        points = beachball.numpy_rtp2xyz(rtp)

        # project to 2D with same projection as used in beachball
        x, y = beachball.project(points, projection=projection).T
        # TODO: adding the DAS condition
        if sta[1].isalpha() and not only_das:
            if polarity == '+':
                ax.plot(
                    x,
                    y,
                    '+',
                    ms=10.0,
                    mew=2.0,
                    mec='r',
                    mfc='none',
                )
            elif polarity == '-':
                ax.plot(
                    x,
                    y,
                    'o',
                    ms=10.0 / np.sqrt(2.0),
                    mew=2.0,
                    mec='b',
                    mfc='none',
                )
            else:
                continue
            ax.text(x + 0.025, y + 0.025, sta, clip_on=True)
        else:
            if polarity == '+':
                ax.plot(
                    x,
                    y,
                    '.',
                    ms=5.0,
                    mew=2.0,
                    mec='none',
                    mfc='r',
                )
            elif polarity == '-':
                ax.plot(
                    x,
                    y,
                    '.',
                    ms=5.0,
                    mew=2.0,
                    mec='none',
                    mfc='b',
                )
            else:
                ax.plot(
                    x,
                    y,
                    '.',
                    ms=5.0,
                    mew=2.0,
                    mec='none',
                    mfc='dimgray',
                )
    if add_info:
        ax.text(
            1.2,
            -0.5,
            f"{source}\nQuality index: {focal_dict['quality_index']}\nnum of station: {focal_dict['num_of_polarity']}\n\
    Strike: {focal_dict['focal_plane']['strike']}\nDip: {focal_dict['focal_plane']['dip']}\n\
    Rake: {focal_dict['focal_plane']['rake']}",
            fontsize=20,
        )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_axis_off()
