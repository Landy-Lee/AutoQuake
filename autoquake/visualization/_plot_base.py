import calendar
import logging
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator


def normalize(x):
    return (x - np.mean(x, axis=-1, keepdims=True)) / np.std(x, axis=-1, keepdims=True)


def formatting(num: str):  # *** why we don't use the :02 in formatting
    """Format single digit numbers with a leading zero."""
    return f'0{num}' if len(num) == 1 else num


def get_total_seconds(dt):
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
    df_seis_phasenet_picks=None,
    df_seis_gamma_picks=None,
    df_das_phasenet_picks=None,
    df_das_gamma_picks=None,
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
