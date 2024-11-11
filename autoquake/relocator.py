from __future__ import annotations

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd


def get_index_table(gamma_reorder_event: Path) -> pd.DataFrame:
    df = pd.read_csv(gamma_reorder_event)
    if 'h3dd_event_index' in df.columns:
        df_table = df.loc[:, ['event_index', 'h3dd_event_index']]
    else:
        df['h3dd_event_index'] = df.index + 1
        df_table = df.loc[:, ['event_index', 'h3dd_event_index']]
    assert isinstance(df_table, pd.DataFrame)
    return df_table


def index_h3dd2gamma(df_table: pd.DataFrame, h3dd_index: int):
    if all(col in df_table.columns for col in ['event_index', 'h3dd_event_index']):
        return df_table[df_table['h3dd_event_index'] == h3dd_index]['event_index'].iloc[
            0
        ]  # noqa: E501
    else:
        raise ValueError(
            "DataFrame must contain 'event_index' and 'h3dd_event_index' columns."
        )


class H3DD:
    def __init__(
        self,
        gamma_event: Path,
        gamma_picks: Path,
        station: Path,
        model_3d: Path,
        event_name='AQ_event',
        weights=[1.0, 1.0, 0.1],
        priori_weight=[1.0, 0.75, 0.5, 0.25, 0.0],
        cut_off_distance_for_dd=3.0,
        inv=2,
        damping_factor=0.0,
        rmscut=1.0e-4,
        max_iter=5,
        constrain_factor=0.0,
        joint_inv_with_single_event_method=1,
        consider_elevation=0,
    ):
        """## Using 3D model for hypoDD.

        ### Args:
            - gamma_event (Path): Path to the gamma event file.
            - gamma_picks (Path): Path to the gamma picks file.
            - model_3d (Path): Path to the 3D model file.
            - event_name (str): Name of the event.
            - weights (list of float): Weighting for P-wave, S-wave, and single event data.
                Example: [1.0, 1.0, 0.1]
            - priori_weight (list of float): A priori weighting for catalog data.
                Example: [1.0, 0.75, 0.5, 0.25, 0.0]
            - cut_off_distance_for_dd (float): Cut-off distance for hypoDD (in km).
            - inv (int): Inversion method. (1 = SVD, 2 = LSQR)
            - damping_factor (float): Damping factor for LSQR.
            - rmscut (float): RMS cut-off threshold.
            - max_iter (int): Maximum number of iterations.
            - constrain_factor (float): Constrain factor.
            - joint_inv_with_single_event_method (int): Joint inversion with single event method.
                (1 = yes, 0 = no)
            - consider_elevation (int): Whether to consider elevation in the model.
                (1 = yes, 0 = no)
        """
        PROJECT_ROOT = Path(__file__).parents[1].resolve()
        self.h3dd_dir = PROJECT_ROOT / 'H3DD'
        self.h3dd_dir.mkdir(parents=True, exist_ok=True)
        self.gamma_event = gamma_event
        self.gamma_picks = gamma_picks
        self.h3dd_station = self._station_h3dd_format(station)
        self.model_3d = self._check_model_3d(model_3d)
        self.event_name = event_name
        self.weights = weights
        self.priori_weight = priori_weight
        self.cut_off_distance_for_dd = cut_off_distance_for_dd
        self.inv = inv
        self.damping_factor = damping_factor
        self.rmscut = rmscut
        self.max_iter = max_iter
        self.constrain_factor = constrain_factor
        self.joint_inv_with_single_event_method = joint_inv_with_single_event_method
        self.consider_elevation = consider_elevation
        self.reorder_event = self.gamma_event.parent / 'gamma_reorder_event.csv'
        self.dout = self.h3dd_dir / f'{event_name}.dat_ch.dout'
        self.hout = self.h3dd_dir / f'{event_name}.dat_ch.hout'

    def _station_h3dd_format(self, station: Path):
        """
        Convert station.csv to h3dd format.
        """
        with open(station) as f:
            first_line = f.readline().strip()
        if first_line.split()[-1] == '21001231':
            print(f'we use {station}')
            return station.name
        df = pd.read_csv(station)
        with open(self.h3dd_dir / 'station.all.select', 'w') as f:
            for _, row in df.iterrows():
                f.write(
                    f"{row['station']} {row['longitude']} {row['latitude']} {row['elevation']} 19010101 21001231\n"
                )
        return 'station.all.select'

    def _check_model_3d(self, model_3d: Path):
        if model_3d.parent != self.h3dd_dir:
            print(f'cp from {model_3d} to {self.h3dd_dir}')
            os.system(f'cp {model_3d} {self.h3dd_dir}')
        return model_3d.name

    def config_h3dd_inp(self, dat_ch: Path):
        if dat_ch.parent != self.h3dd_dir:
            os.system(f'cp {dat_ch} {self.h3dd_dir}')
        with open(self.h3dd_dir / 'h3dd.inp', 'w') as f:
            f.write('*1. input catalog data\n')
            f.write(f'{dat_ch.name}\n')
            f.write('*2. station information file\n')
            f.write(f'{self.h3dd_station}\n')
            f.write('*3. 3d velocity model\n')
            f.write(f'{self.model_3d}\n')
            f.write('*4. weighting for p wave, s wave, and single event data\n')
            f.write('*   wp  ws  wsingle\n')
            f.write(
                f"{' '*4}{self.weights[0]:<4}{self.weights[1]:<4}{self.weights[2]:>4}\n"
            )
            f.write('*5. a priori weighting for catalog data\n')
            f.write('*   0      1      2      3      4\n')
            f.write(
                f"{' '*4}{self.priori_weight[0]:<5}{self.priori_weight[1]:<7}{self.priori_weight[2]:<7}{self.priori_weight[3]:<7}{self.priori_weight[4]:>3}\n"
            )
            f.write('*6. cut off distance for D-D method (km)\n')
            f.write(f"{' '*4}{self.cut_off_distance_for_dd}\n")
            f.write('*7. inv (1=SVD 2=LSQR)\n')
            f.write(f"{' '*4}{self.inv}\n")
            f.write('*8. damping factor (Only work if inv=2)\n')
            f.write(f"{' '*4}{self.damping_factor}\n")
            f.write('*9. rmscut (sec)\n')
            f.write(f"{' '*4}{self.rmscut}\n")
            f.write('*10. maximum interation times\n')
            f.write(f"{' '*4}{self.max_iter}\n")
            f.write('*11. constrain factor\n')
            f.write(f"{' '*4}{self.constrain_factor}\n")
            f.write('*12. joint inversion with single event method (1=yes 0=no)\n')
            f.write(f"{' '*4}{self.joint_inv_with_single_event_method}\n")
            f.write('*13. consider elevation or not (1=yes 0=no)\n')
            f.write(f"{' '*4}{self.consider_elevation}\n")

    def _gamma_reorder(self, gamma_reorder_event=None) -> pd.DataFrame:
        df = pd.read_csv(self.gamma_event)
        df['time'] = pd.to_datetime(df['time'])
        df_sort = df.sort_values(by='time')
        df_sort = df_sort.reset_index(drop=True)
        df_sort['h3dd_event_index'] = df_sort.index + 1

        if gamma_reorder_event is None:
            gamma_reorder_event = self.gamma_event.parent / 'gamma_reorder_event.csv'
        df_sort.to_csv(gamma_reorder_event, index=False)
        return df_sort

    def _gamma_preprocess(
        self, df_event: pd.DataFrame, df_picks: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        df_event['time'] = pd.to_datetime(df_event['time'])
        df_event['ymd'] = df_event['time'].dt.strftime('%Y%m%d')
        df_event['hour'] = df_event['time'].dt.hour
        df_event['minute'] = df_event['time'].dt.minute
        df_event['seconds'] = (
            df_event['time'].dt.second + df_event['time'].dt.microsecond / 1_000_000
        )
        df_event['lon_int'] = df_event['longitude'].apply(lambda x: int(x))
        df_event['lon_deg'] = (
            df_event['longitude'].apply(lambda x: float(x)) - df_event['lon_int']
        ) * 60
        df_event['lat_int'] = df_event['latitude'].apply(lambda x: int(x))
        df_event['lat_deg'] = (
            df_event['latitude'].apply(lambda x: float(x)) - df_event['lat_int']
        ) * 60
        df_event['depth'] = df_event['depth_km'].round(2)

        df_picks['phase_time'] = pd.to_datetime(df_picks['phase_time'])
        df_picks['minute'] = df_picks['phase_time'].dt.minute
        df_picks['seconds'] = (
            df_picks['phase_time'].dt.second
            + df_picks['phase_time'].dt.microsecond / 1_000_000
        )
        return df_event, df_picks

    def get_gamma(
        self, output_file: Path, df_event: pd.DataFrame, df_picks: pd.DataFrame
    ):
        event_indices = [event for event in df_event['event_index'] if event != -1]
        with open(output_file, 'w') as r:
            for i in event_indices:
                row = df_event[df_event['event_index'] == i]
                # logging.info(f'***Event {i}***')
                r.write(
                    f"{row['ymd'].iloc[0]:>9}{row['hour'].iloc[0]:>2}{row['minute'].iloc[0]:>2}{row['seconds'].iloc[0]:>6.2f}{row['lat_int'].iloc[0]:2}{row['lat_deg'].iloc[0]:0>5.2f}{row['lon_int'].iloc[0]:3}{row['lon_deg'].iloc[0]:0>5.2f}{row['depth'].iloc[0]:>6.2f}\n"
                )
                # check_box = []
                for _, pick_row in df_picks[df_picks['event_index'] == i].iterrows():
                    # TODO: Check the time format
                    if row['minute'].iloc[0] == 59 and pick_row['minute'] == 0:
                        wmm = 60
                    else:
                        wmm = pick_row['minute']
                    weight = '1.00'
                    # TODO: DAS
                    # check_pattern = f"{pick_row['phase_type']}_{pick_row['minute']}_{pick_row['seconds']}"
                    # if check_pattern not in check_box:
                    #     check_box.append(check_pattern)
                    # else:
                    #     logging.info(f"{pick_row['station_id']} has same {pick_row['phase_type']} arrival time")
                    #     continue
                    if pick_row['phase_type'] == 'P':
                        r.write(
                            f"{' ':1}{pick_row['station_id']:<4}{'0.0':>6}{'0':>4}{'0':>4}{wmm:>4}{pick_row['seconds']:>6.2f}{'0.01':>5}{weight:>5}{'0.00':>6}{'0.00':>5}{'0.00':>5}\n"
                        )
                    else:
                        r.write(
                            f"{' ':1}{pick_row['station_id']:<4}{'0.0':>6}{'0':>4}{'0':>4}{wmm:>4}{'0.00':>6}{'0.00':>5}{'0.00':>5}{pick_row['seconds']:>6.2f}{'0.01':>5}{weight:>5}\n"
                        )

    def process_chunk(self, chunk, file_num, df_picks):
        """Process a single chunk and output the result to a file."""
        # Preprocess the chunk
        df_event, df_picks = self._gamma_preprocess(df_event=chunk, df_picks=df_picks)

        # Generate the output file path
        output_file = self.h3dd_dir / f'{self.event_name}_{file_num}.dat_ch'

        # Call the get_gamma method to process and save the chunk
        self.get_gamma(output_file=output_file, df_event=df_event, df_picks=df_picks)

    def process_in_parallel(self, df_event, df_picks, chunk_size):
        """Split the DataFrame into chunks and process them in parallel."""
        # Create a list of chunks to process
        chunks = [
            (df_event.iloc[i : i + chunk_size], file_num, df_picks)
            for file_num, i in enumerate(range(0, len(df_event), chunk_size))
        ]
        self.file_num = len(chunks)
        # Use ThreadPoolExecutor to process chunks in parallel
        with ThreadPoolExecutor() as executor:
            # Submit each chunk to be processed in a separate thread
            futures = [
                executor.submit(self.process_chunk, chunk, file_num, df_picks)
                for chunk, file_num, df_picks in chunks
            ]

            # Ensure all threads complete and raise exceptions if any
            for future in futures:
                future.result()

    def gamma2h3dd(self, chunk_size=4000):
        """
        Convert ordered event and picks into h3dd format dat_ch.
        """
        df_event = self._gamma_reorder()
        df_picks = pd.read_csv(self.gamma_picks)
        self.h3dd_dir.mkdir(parents=True, exist_ok=True)

        # split the event if exceed 4000
        if len(df_event) > 4000:
            self.process_in_parallel(df_event, df_picks, chunk_size)
        else:
            output_file = self.h3dd_dir / f'{self.event_name}.dat_ch'
            df_event, df_picks = self._gamma_preprocess(
                df_event=df_event, df_picks=df_picks
            )
            self.get_gamma(
                output_file=output_file, df_event=df_event, df_picks=df_picks
            )
            self.file_num = 1

    def run_h3dd(self):
        """
        Running 3D HypoDD.
        """
        self.gamma2h3dd()
        if self.file_num > 1:
            for i in range(self.file_num):
                self.config_h3dd_inp(
                    dat_ch=self.h3dd_dir / f'{self.event_name}_{i}.dat_ch'
                )

                working_dir = self.h3dd_dir
                # Open the input file safely using a context manager
                with open(working_dir / 'h3dd.inp') as inp_file:
                    # Run the executable with input from the file
                    result = subprocess.run(
                        ['./h3dd'],
                        stdin=inp_file,  # Redirect input from the file
                        text=True,
                        cwd=working_dir,
                    )

                if result.returncode != 0:
                    print('Error occurred during h3dd execution.')

            # concat if file_num > 1
            self.post_h3dd()
        else:
            self.config_h3dd_inp(dat_ch=self.h3dd_dir / f'{self.event_name}.dat_ch')
            working_dir = self.h3dd_dir
            # Open the input file safely using a context manager
            with open(working_dir / 'h3dd.inp') as inp_file:
                # Run the executable with input from the file
                result = subprocess.run(
                    ['./h3dd'],
                    stdin=inp_file,  # Redirect input from the file
                    text=True,
                    cwd=working_dir,
                )

            if result.returncode != 0:
                print('Error occurred during h3dd execution.')

        os.system(f'cp {self.hout} {self.gamma_event.parent}')
        os.system(f'cp {self.dout} {self.gamma_event.parent}')

    def just_run(self, dat_ch: Path):
        self.config_h3dd_inp(dat_ch=dat_ch)

        working_dir = self.h3dd_dir

        # Open the input file safely using a context manager
        with open(working_dir / 'h3dd.inp') as inp_file:
            # Run the executable with input from the file
            result = subprocess.run(
                ['./h3dd'],
                stdin=inp_file,  # Redirect input from the file
                text=True,
                cwd=working_dir,
            )

        if result.returncode != 0:
            print('Error occurred during h3dd execution.')

    def post_h3dd(self):
        """
        concat the dout and hout once the file_num > 1.
        """
        if self.file_num > 1:
            for ftype in ['dat_ch.hout', 'dat_ch.dout']:
                output_file = self.h3dd_dir / f'{self.event_name}.{ftype}'
                with open(output_file, 'w') as outfile:
                    for i in range(self.file_num):
                        fname = self.h3dd_dir / f'{self.event_name}_{i}.{ftype}'
                        with open(fname) as infile:
                            # NOTE: Using shutil.copyfileobj() for large file.
                            outfile.write(infile.read())

    @staticmethod
    def pol_mag_to_dout(
        ori_dout: Path,
        gamma_reorder_event: Path,
        polarity_picks: Path,
        magnitude_events: Path,
        magnitude_picks: Path,
        output_path: Path,
    ):
        """
        Combining polarity and magnitude information into dout.
        """
        df_table = get_index_table(gamma_reorder_event=gamma_reorder_event)

        df_pol = pd.read_csv(polarity_picks)
        df_mag_event = pd.read_csv(magnitude_events)
        df_mag_pick = pd.read_csv(magnitude_picks)
        output_dout = output_path / f'{ori_dout.name}'
        with open(output_dout, 'w') as fo:
            with open(ori_dout) as r:
                lines = r.readlines()
            h3dd_event_index = 0
            for line in lines:
                if line.strip().split()[-1] == '3DD':
                    h3dd_event_index += 1
                    event_mag = round(
                        df_mag_event[
                            df_mag_event['h3dd_event_index'] == h3dd_event_index
                        ]['magnitude'].iloc[0],
                        2,
                    )

                    fo.write(f'{line[:40]}{event_mag:4.2f}{line[44:]}')
                elif line[35:39] == '1.00':
                    station = line[:5].strip()
                    sta_mag = round(
                        df_mag_pick[
                            (df_mag_pick['h3dd_event_index'] == h3dd_event_index)
                            & (df_mag_pick['station'] == station)
                        ]['magnitude'].iloc[0],
                        2,
                    )
                    polarity = df_pol[
                        (
                            df_pol['event_index']
                            == index_h3dd2gamma(df_table, h3dd_event_index)
                        )
                        & (df_pol['station_id'] == station)
                    ]['polarity'].iloc[0]
                    if polarity == 'U':
                        polarity = '+'
                    elif polarity == 'D':
                        polarity = '-'
                    else:
                        polarity = ' '

                    fo.write(
                        f'{line[:19]}{polarity} {line[21:].strip()} 0.00 0.00 0.00 {sta_mag:4.2f} 0   0.0\n'
                    )
        return ori_dout.name

    @staticmethod
    def pol_to_dout(
        ori_dout: Path,
        gamma_reorder_event: Path,
        polarity_picks: Path,
        output_path: Path,
    ):
        """
        Combining polarity and magnitude information into dout.
        """
        df_table = get_index_table(gamma_reorder_event=gamma_reorder_event)

        df_pol = pd.read_csv(polarity_picks)
        output_dout = output_path / f'{ori_dout.name}'
        with open(output_dout, 'w') as fo:
            with open(ori_dout) as r:
                lines = r.readlines()
            h3dd_event_index = 0
            for line in lines:
                if line.strip().split()[-1] == '3DD':
                    h3dd_event_index += 1
                    fo.write(line)
                elif line[35:39] == '1.00':
                    station = line[:5].strip()
                    polarity = df_pol[
                        (
                            df_pol['event_index']
                            == index_h3dd2gamma(df_table, h3dd_event_index)
                        )
                        & (df_pol['station_id'] == station)
                    ]['polarity'].iloc[0]
                    if polarity == 'U':
                        polarity = '+'
                    elif polarity == 'D':
                        polarity = '-'
                    else:
                        polarity = ' '

                    fo.write(
                        f'{line[:19]}{polarity} {line[21:].strip()} 0.00 0.00 0.00 0.00 0   0.0\n'
                    )

        return ori_dout.name
