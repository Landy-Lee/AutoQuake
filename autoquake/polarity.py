from __future__ import annotations

import logging
import multiprocessing as mp
import os
import warnings
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import onnxruntime as ort
import pandas as pd
from obspy import UTCDateTime, read

diting_model = (
    Path(__file__).parents[1].resolve() / 'focal_model' / 'DiTingMotionJul.onnx'
)

warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '2'  # Adjust based on your system
os.environ['MKL_NUM_THREADS'] = '2'
session_options = ort.SessionOptions()
session_options.intra_op_num_threads = 2
session_options.inter_op_num_threads = 1


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


class DitingMotion:
    def __init__(
        self,
        gamma_picks: Path,
        model_path=diting_model,
        output_dir=None,
        sac_parent_dir=None,
        h5_parent_dir=None,
        interval=300,
        sampling_rate=100,
    ):
        self.gamma_picks = gamma_picks
        self.model_path = model_path
        self.output_dir = output_dir
        self.sac_parent_dir = sac_parent_dir
        self.h5_parent_dir = h5_parent_dir
        self.interval = interval
        self.sampling_rate = sampling_rate
        self.indices = self._get_indices()

    def _get_indices(self):
        df = pd.read_csv(self.gamma_picks)
        return [x for x in set(df['event_index']) if x != -1]

    def seis_get_data(self, sta_name: str, p_arrival_: str, time_window=0.64):
        ymd = time_formatting(p_arrival_)
        p_arrival = UTCDateTime(p_arrival_)
        if self.sac_parent_dir is not None:
            data_path = self.sac_parent_dir / ymd  # data
        else:
            raise ValueError('Please provide sac_parent_dir')
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
    def das_for_model(self, data, total_seconds):
        event_index = int((total_seconds % self.interval) * self.sampling_rate)
        return data[
            event_index - int(0.64 * self.sampling_rate) : event_index
            + int(0.64 * self.sampling_rate)
        ]

    def das_get_data(self, sta_name: str, p_arrival: str):
        total_seconds = get_total_seconds(pd.to_datetime(p_arrival))
        index = int(total_seconds // self.interval)
        window = f'{self.interval*index}_{self.interval*(index+1)}.h5'
        if self.h5_parent_dir is not None:
            try:
                file = list(self.h5_parent_dir.glob(f'*{window}'))[0]
            except IndexError:
                logging.info(f'File not found for window {window}')
                return []
        else:
            raise ValueError('Please provide h5_parent_dir')

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
        data = self.das_for_model(data, total_seconds)
        return data

    def diting_motion(self, data, row, motion_model):
        # create zeros array
        motion_input = np.zeros([1, 128, 2], dtype=np.float32)
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
                diff_data = np.diff(
                    motion_input[0, 64:, 0]
                )  # difference between 64: data.
                diff_sign_data = np.sign(diff_data)
                motion_input[0, 65:, 1] = diff_sign_data[:]

                # model prediction
                # logging.info(f"starting predict_chunk{i}_event_{counter}")
                ort_inputs = {motion_model.get_inputs()[0].name: motion_input}
                pred_res = motion_model.run(None, ort_inputs)
                # pred_res = motion_model.predict(motion_input)
                pred_fmp = (
                    pred_res[0]  # T0D0
                    + pred_res[1]  # T0D1
                    + pred_res[2]  # T0D2
                    + pred_res[3]  # T0D3
                ) / 4
                pred_cla = (
                    pred_res[4]  # T1D0
                    + pred_res[5]  # T1D1
                    + pred_res[6]  # T1D2
                    + pred_res[7]  # T1D3
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

    def process_row(self, row, motion_model) -> pd.DataFrame:
        if row.station_id[1].isdigit():
            data = self.das_get_data(
                sta_name=row.station_id,
                p_arrival=row.phase_time,
            )
        elif row.station_id[1].isalpha():
            data = self.seis_get_data(
                sta_name=row.station_id, p_arrival_=row.phase_time
            )
        if data == []:
            logging.info(f'empty data exist in {row.station_id}')

        polarity = self.diting_motion(data, row, motion_model)
        row['polarity'] = polarity
        return row

    def predict(self, event_index) -> list:
        # Read the gamma_picks CSV file into a DataFrame
        # print(f"predict_{event_index}")
        df_picks = pd.read_csv(self.gamma_picks)

        # Filter DataFrame for the specific event_index and phase_type 'P'
        df_selected_picks = df_picks[
            (df_picks['event_index'] == event_index) & (df_picks['phase_type'] == 'P')
        ]
        # Iterate through the selected picks
        logging.info(f'event index: {event_index} start processing')
        ort_session = ort.InferenceSession(
            self.model_path, sess_options=session_options
        )
        processed_rows = []
        for _, row in df_selected_picks.iterrows():
            df_row = self.process_row(row=row, motion_model=ort_session)
            if not df_row.empty:
                processed_rows.append(df_row)
        logging.info(f'event index: {event_index} processing over')
        return processed_rows

    def run_parallel_predict(self, processes=10):
        if self.output_dir is None:
            self.output_dir = Path(__file__).parents[1].resolve() / 'diting_result'
            self.output_dir.mkdir(parents=True, exist_ok=True)
        output_csv = self.output_dir / 'gamma_events_polarity.csv'
        # if output_csv.exists():
        #     print(f'remove {output_csv}')
        #     output_csv.unlink()
        #     logging.info('remove the current csv file already!')

        # Use a process pool to parallelize the work
        with mp.Pool(processes=processes) as pool:
            results = pool.starmap(
                self.predict, [(event_index,) for event_index in self.indices]
            )

        all_picks = [item for sublist in results for item in sublist]
        if all_picks:
            df__result = pd.DataFrame(all_picks)
            df__result.to_csv(output_csv, index=False)
