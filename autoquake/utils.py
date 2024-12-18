import multiprocessing as mp

import pandas as pd

from autoquake.visualization._plot_base import utc_to_timestamp


def dmm_trans(coord: str):
    """
    Convert degree and minute format (e.g., '2523.47') to decimal degrees.
    :param coord: Coordinate in degree and minute format as a string.
    :return: Decimal degrees as a float.
    """
    try:
        # Split into degrees and minutes
        degrees = int(coord[:-5])  # Extract the degree part (all but last 5 chars)
        minutes = float(coord[-5:])  # Extract the minute part (last 5 chars)

        # Convert to decimal degrees
        return degrees + (minutes / 60)
    except (ValueError, IndexError):
        raise ValueError(f'Invalid coordinate format: {coord}')


def comparing_picks(df_gamma_picks, time_array, i, tol=3):
    print(f'event_{i}')
    df = df_gamma_picks[df_gamma_picks['event_index'] == i]
    if df.empty:
        return []
    df = df[df['station_id'].map(lambda x: x[1].isdigit())]
    std_time = df['phase_time'].apply(utc_to_timestamp).mean()
    index = [i for i, t in enumerate(time_array) if abs(std_time - t) <= tol]
    return index


def process_event_wrapper(args):
    """
    Process a single event index to extract matching rows and append the event_index.
    """
    event_index, df_gamma_picks, df_phasenet, time_array, tol = args
    index = comparing_picks(df_gamma_picks, time_array, event_index, tol)
    if not index:
        return None
    df_target = df_phasenet.loc[index].copy()
    df_target['event_index'] = event_index
    return df_target


def pseudo_picks_generator(
    phasenet_picks, gamma_picks, das_station, das_station_20, pseudo_gamma_picks, tol=3
):
    """## Searching for phasenet_das picks that not used to associate into associate picks by time dependency.
    Example
    df_result = pseudo_picks_generator(
        phasenet_picks=phasenet_picks,
        gamma_picks=gamma_picks,
        pseudo_gamma_picks=pseudo_gamma_picks,
        das_station=Path('/home/patrick/Work/EQNet/tests/hualien_0403/station_das.csv'),
        das_station_20=Path('/home/patrick/Work/Hualien0403/stations/das_20.csv'),
    )
    """
    df_das_sta = pd.read_csv(das_station)
    df_das_20 = pd.read_csv(das_station_20)
    sta_set = set(df_das_sta['station']) - set(df_das_20['station'])
    df_phasenet = pd.read_csv(phasenet_picks)
    df_phasenet = (
        df_phasenet[df_phasenet['station_id'].isin(sta_set)]
        .copy()
        .reset_index(drop=True)
    )
    time_array = df_phasenet['phase_time'].apply(utc_to_timestamp).to_numpy()
    df_gamma_picks = pd.read_csv(gamma_picks)
    event_indices = set(x for x in df_gamma_picks['event_index'] if x != -1)
    # another function
    # Prepare arguments for multiprocessing
    args = [(i, df_gamma_picks, df_phasenet, time_array, tol) for i in event_indices]

    # Use multiprocessing Pool
    with mp.Pool(processes=40) as pool:
        results = pool.map(process_event_wrapper, args)

    # Combine results into a DataFrame
    collected_rows = pd.concat(
        [res for res in results if res is not None], ignore_index=True
    )

    collected_rows.drop(columns={'channel_index'}, inplace=True)
    collected_rows['gamma_score'] = 999

    # Concatenate collected_rows with df_gamma_picks
    result_df = pd.concat([df_gamma_picks, collected_rows], ignore_index=True)

    # Optionally save to file
    result_df.to_csv(pseudo_gamma_picks, index=False)

    return result_df
