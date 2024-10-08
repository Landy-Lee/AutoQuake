import calendar
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


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


def txt_preprocessor(df: pd.DataFrame) -> pd.DataFrame:
    """
    Distinguish h3dd and gafocal format through YYYYMMDD & YYYY/MM/DD.
    """
    if len(df[0].iloc[0]) == 8:  # h3dd
        df[1] = df[1].apply(check_hms_h3dd)
        df['time'] = (
            df[0].apply(lambda x: f'{x[:4]}-{x[4:6]}-{x[6:8]}')
            + 'T'
            + df[1].apply(lambda x: f'{x[:2]}:{x[2:4]}:{x[4:6]}.{x[6:8]}')
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
        df = txt_preprocessor(df)
    timestamp = df['time'].apply(utc_to_timestamp).to_numpy()
    return df, timestamp
