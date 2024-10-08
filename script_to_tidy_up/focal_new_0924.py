# %%
import logging
import multiprocessing as mp
import os
import warnings
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K  # for interacting with tensorflow
from obspy import UTCDateTime, read

# Suppress warnings
warnings.filterwarnings('ignore')

# Set environment variables to control resource usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'  # Adjust based on your system
os.environ['TF_NUM_INTRAOP_THREADS'] = '2'
os.environ['TF_NUM_INTEROP_THREADS'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Clear any previous TensorFlow sessions to free up memory
K.clear_session()


def formatting(num):
    """Format number to have leading zero if needed."""
    return f'0{num}' if len(num) == 1 else num


def time_formatting(time: str):
    date_obj = datetime.strptime(time, '%Y-%m-%dT%H:%M:%S.%f')

    # Format the datetime object to 'YYYYMMDD'
    return date_obj.strftime('%Y%m%d')


def count_files_in_dir(directory_path: Path):
    if not directory_path.is_dir():
        raise ValueError(f'The path {directory_path} is not a directory.')
    return sum(1 for item in directory_path.iterdir() if item.is_file())


def seis_get_data(
    sac_parent_dir: Path, sta_name: str, p_arrival: str, time_window=0.64
):
    ymd = time_formatting(p_arrival)
    p_arrival = UTCDateTime(p_arrival)
    data_path = sac_parent_dir / ymd / 'data_final'  # data
    try:
        sac_file = list(data_path.glob(f'*{sta_name}*Z*'))[0]
    except Exception as e:
        logging.info(f'Error: {sta_name}_{e} when p_arrival: {p_arrival}')
        return []
    try:
        st = read(sac_file)
        st.detrend('demean')
        st.detrend('linear')
        st.taper(0.001)
        st.resample(sampling_rate=100)
        starttime_trim = p_arrival - time_window
        endtime_trim = p_arrival + time_window
        st.trim(starttime=starttime_trim, endtime=endtime_trim)
        data = st[0].data[0:128]
        return data
    except Exception as e:
        logging.info(f'Error exist during process {sac_file}: {e}')
        return []


# for DAS
def convert_channel_index(sta_name: str) -> int:
    if sta_name[:1] == 'A':
        return int(sta_name[1:])
    elif sta_name[:1] == 'B':
        return int(f'1{sta_name[1:]}')
    else:
        raise ValueError('wrong format warning: please append the condition')


def get_total_seconds(dt):
    return (dt - dt.normalize()).total_seconds()


def das_demean(data):
    return data - np.mean(data)


def das_detrend(data):
    from scipy.signal import detrend

    return detrend(data)


def das_for_model(data, total_seconds, interval=300, sampling_rate=100):
    event_index = int((total_seconds % interval) * sampling_rate)
    return data[
        event_index - int(0.64 * sampling_rate) : event_index
        + int(0.64 * sampling_rate)
    ]


def das_get_data(hdf5_parent_dir: Path, sta_name: str, p_arrival, interval=300):
    total_seconds = get_total_seconds(pd.to_datetime(p_arrival))
    index = int(total_seconds // interval)
    window = f'{interval*index}_{interval*(index+1)}.h5'
    try:
        file = list(hdf5_parent_dir.glob(f'*{window}'))[0]
    except IndexError:
        logging.info(f'File not found for window {window}')
        return []

    channel_index = convert_channel_index(sta_name)
    try:
        with h5py.File(file, 'r') as fp:
            ds = fp['data']
            # xxx = dir(ds)
            data = ds[channel_index]
    except Exception as e:
        logging.info(f'Error reading {file}: {e}')
        return []
    data = das_demean(data)
    data = das_detrend(data)
    data = das_for_model(data, total_seconds)
    return data


def diting_motion(data, row, motion_model):
    # create zeros array
    motion_input = np.zeros([1, 128, 2])
    try:
        motion_input[0, :, 0] = data
    except Exception as e:
        logging.info(f'Error: {e} -> row: {row}')
        logging.info(f'data: {data}')
    if np.max(motion_input[0, :, 0]) != 0:
        motion_input[0, :, 0] -= np.mean(
            motion_input[0, :, 0]
        )  # waveform demean -> centralization
        norm_factor = np.std(motion_input[0, :, 0])  # standard deviation
        if norm_factor != 0:
            motion_input[0, :, 0] /= norm_factor  # normalization
            diff_data = np.diff(motion_input[0, 64:, 0])  # difference between 64: data.
            diff_sign_data = np.sign(diff_data)
            motion_input[0, 65:, 1] = diff_sign_data[:]

            # model prediction
            # logging.info(f"starting predict_chunk{i}_event_{counter}")
            pred_res = motion_model.predict(motion_input)
            pred_fmp = (
                pred_res['T0D0']
                + pred_res['T0D1']
                + pred_res['T0D2']
                + pred_res['T0D3']
            ) / 4
            pred_cla = (
                pred_res['T1D0']
                + pred_res['T1D1']
                + pred_res['T1D2']
                + pred_res['T1D3']
            ) / 4
            # logging.info(f"predict done_chunk{i}_event_{counter}")
            # logging.info(pred_fmp, pred_cla)
            if np.argmax(pred_fmp[0, :]) == 1:
                polarity = 'D'
                symbol = '-'
                if np.argmax(pred_cla[0, :]) == 0:
                    sharpness = 'I'
                elif np.argmax(pred_cla[0, :]) == 1:
                    sharpness = 'E'
                else:
                    sharpness = 'x'
            elif np.argmax(pred_fmp[0, :]) == 0:
                polarity = 'U'
                symbol = '+'
                if np.argmax(pred_cla[0, :]) == 0:
                    sharpness = 'I'
                elif np.argmax(pred_cla[0, :]) == 1:
                    sharpness = 'E'
                else:
                    sharpness = 'x'
            else:
                polarity = 'x'
                symbol = ' '
                if np.argmax(pred_cla[0, :]) == 0:
                    sharpness = 'I'
                elif np.argmax(pred_cla[0, :]) == 1:
                    sharpness = 'E'
                else:
                    sharpness = 'x'
    return polarity


def process_row(row, motion_model, hdf5_parent_dir, sac_parent_dir):
    if row.station_id[1].isdigit():
        data = das_get_data(
            hdf5_parent_dir=hdf5_parent_dir,
            sta_name=row.station_id,
            p_arrival=row.phase_time,
        )
    elif row.station_id[1].isalpha():
        data = seis_get_data(
            sac_parent_dir=sac_parent_dir,
            sta_name=row.station_id,
            p_arrival=row.phase_time,
        )
    if data == []:
        logging.info(f'empty data exist in {row.station_id}')
        return

    polarity = diting_motion(data, row, motion_model)
    row['polarity'] = polarity
    return row


def predict(
    event_index,
    gamma_picks,
    model_path,
    lock,
    output_csv,
    hdf5_parent_dir,
    sac_parent_dir,
):
    # Read the gamma_picks CSV file into a DataFrame
    df_picks = pd.read_csv(gamma_picks)

    # Filter DataFrame for the specific event_index and phase_type 'P'
    df_selected_picks = df_picks[
        (df_picks['event_index'] == event_index) & (df_picks['phase_type'] == 'P')
    ]
    # Iterate through the selected picks
    logging.info(f'event index: {event_index} start processing')
    motion_model = tf.keras.models.load_model(model_path, compile=False)
    processed_rows = []
    for _, row in df_selected_picks.iterrows():
        processed_row = process_row(
            row=row,
            hdf5_parent_dir=hdf5_parent_dir,
            sac_parent_dir=sac_parent_dir,
            motion_model=motion_model,
        )
        if processed_row is not None:
            processed_rows.append(processed_row)
    logging.info(f'event index: {event_index} processing over')
    if processed_rows:
        with lock:  # Ensure thread-safe access to the file
            write_header = not os.path.exists(output_csv)
            pd.DataFrame(processed_rows).to_csv(
                output_csv, mode='a', header=write_header, index=False
            )
    logging.info(f'event index: {event_index} write into the csv')
    return processed_rows


def run_parallel_predict(
    event_indices,
    gamma_picks,
    model_path,
    output_dir,
    hdf5_parent_dir=None,
    sac_parent_dir=None,
):
    output_csv = output_dir / 'gamma_events_polarity.csv'
    if output_csv.exists():
        print(f'remove {output_csv}')
        output_csv.unlink()
        logging.info('remove the current csv file already!')
    # Manager to handle shared memory
    with mp.Manager() as manager:
        # shared_list = manager.list()  # Shared list to collect rows from all processes
        lock = manager.Lock()  # Lock to ensure thread-safe access to the shared list

        # Define a function to handle the append operation with locking mechanism
        # def collect_results(result):
        #     with lock:
        #         shared_list.extend(result)

        # Use a process pool to parallelize the work
        with mp.Pool(processes=2) as pool:
            # results = []
            for event_index in event_indices:
                pool.apply_async(
                    predict,
                    args=(
                        event_index,
                        gamma_picks,
                        model_path,
                        lock,
                        output_csv,
                        hdf5_parent_dir,
                        sac_parent_dir,
                    ),
                    # callback=collect_results  # Add results to the shared list once a process completes
                )
                # results.append(result)
            # Close the pool and wait for all processes to finish
            pool.close()
            pool.join()

        # Convert the shared list into a DataFrame
        # result_df = pd.DataFrame(list(shared_list))

        # Save the result DataFrame to a CSV file
        # result_df.to_csv(output_csv, index=False)


if __name__ == '__main__':
    gamma_events = Path(
        '/data2/patrick/Hualien0403/GaMMA/gamma_test/synthetic_main_eq/gamma_events.csv'
    )
    gamma_picks = Path(
        '/data2/patrick/Hualien0403/GaMMA/gamma_test/synthetic_main_eq/gamma_picks.csv'
    )
    sac_parent_dir = Path('/data2/patrick/Hualien0403/Dataset/')  # merging sac data
    model_path = Path(
        '/home/patrick/Work/AutoQuake_Focal_pamicoding/DiTing-FOCALFLOW/models/DiTingMotionJul.hdf5'
    )  # polarity model
    hdf5_parent_dir = Path('/raid4/DAS_data/iDAS_MiDAS/hdf5/20240402_hdf5')
    output_dir = Path('/data2/patrick/Hualien0403/GaMMA/gamma_test/synthetic_main_eq/')

    df_counter = pd.read_csv(gamma_events)
    event_indices = set(
        df_counter['event_index']
    )  # Replace with your actual event indices

    logging.basicConfig(
        filename='logger.log',
        format='%(asctime)s %(message)s',
        filemode='w',
        level=logging.INFO,
    )
    # Run the parallel prediction process
    run_parallel_predict(
        event_indices=event_indices,
        gamma_picks=gamma_picks,
        hdf5_parent_dir=hdf5_parent_dir,
        sac_parent_dir=sac_parent_dir,
        model_path=model_path,
        output_dir=output_dir,
    )
