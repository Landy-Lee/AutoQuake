import calendar
import logging
import math
import multiprocessing as mp
from pathlib import Path
from typing import Tuple

import pandas as pd
from obspy import UTCDateTime, read
from obspy.io.sac.sacpz import attach_paz
from .utils import degree_trans

pre_filt = (0.1, 0.5, 30, 35)

wa_simulate = {
    'poles': [(-6.28318 + 4.71239j), (-6.28318 - 4.71239j)],
    'zeros': [0j, 0j],
    'sensitivity': 1.0,
    'gain': 2800.0,
}


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

    return f'{year}-{month:02}-{day:02}T{hour:02}:{min:02}:{sec}'


def get_phase_utc(year, month, day, hour, min, sec, phase_min, phase_sec):
    """
    This function is to check the time of the phase.
    """
    if phase_min < min:  # example: 08:00:05 07:59:59
        hour += 1
        return check_time(year, month, day, hour, phase_min, phase_sec)
    else:
        return check_time(year, month, day, hour, phase_min, phase_sec)


class Magnitude:
    """
    This class is to estimate the magnitude thorugh h3dd format.

    example:
    mag = Magnitude(
        dout_file=Path('/home/patrick/Work/AutoQuake/test_format/test.dout'),
        station_info=Path('/home/patrick/Work/EQNet/tests/hualien_0403/station_seis.csv'),
        pz_path=Path('/data/share/for_patrick/PZ_test'),
        sac_parent_dir=Path('/data2/patrick/Hualien0403/Dataset/20240401/')
    )
    """

    def __init__(
        self,
        dout_file: Path,
        station_info: Path,
        sac_parent_dir: Path,
        pz_path: Path,
    ):
        self.dout_file = dout_file
        self.station_info = station_info
        self.sac_parent_dir = sac_parent_dir
        self.pz_path = pz_path

    def process_h3dd(self):
        """
        Processing h3dd into mag ready format
        """
        with open(self.dout_file) as f:
            lines = f.readlines()

        mag_event_header = [
            'year',
            'month',
            'day',
            'utctime',
            'total_seconds',
            'longitude',
            'latitude',
            'depth',
            'event_index',
        ]
        mag_picks_header = [
            'station',
            'phase_time',
            'total_seconds',
            'phase_type',
            'dist',
            'azimuth',
            'takeoff_angle',
            'elevation',
            'event_index',
        ]
        df_station = pd.read_csv(
            self.station_info,
            dtype={
                'station': 'str',
                'longitude': 'float',
                'latitude': 'float',
                'elevation': 'float',
            },
        )
        mag_event_list = []
        mag_picks_list = []
        event_index = 0
        for line in lines:
            if line.strip()[0].isdigit():
                event_index += 1
                year = int(line[1:5].strip())
                month = int(line[5:7].strip())
                day = int(line[7:9].strip())
                hour = int(line[9:11].strip())
                min = int(line[11:13].strip())
                sec = float(line[13:19].strip())
                total_seconds = hour * 3600 + min * 60 + sec
                utctime = f'{year:4}-{month:02}-{day:02}T{hour:02}:{min:02}:{sec:05.2f}'
                lat_part = line[19:26].strip()
                lon_part = line[26:34].strip()
                event_lon = round(degree_trans(lon_part), 3)
                event_lat = round(degree_trans(lat_part), 3)
                depth = line[34:40].strip()
                mag_event_list.append(
                    [
                        year,
                        month,
                        day,
                        utctime,
                        total_seconds,
                        event_lon,
                        event_lat,
                        depth,
                        event_index,
                    ]
                )
            else:
                station = line[1:5].strip()
                dist = line[5:11]
                azi = int(line[12:15].strip())
                toa = int(line[16:19].strip())
                phase_min = int(line[20:23].strip())
                p_weight = line[35:39].strip()
                s_weight = line[51:55].strip()
                elevation = (
                    df_station[df_station['station'] == station].iloc[0].elevation / 1000
                )

                if p_weight == '1.00':
                    phase_sec = float(line[23:29].strip())
                    phase_type = 'P'

                elif s_weight == '1.00':
                    phase_sec = float(line[40:45].strip())
                    phase_type = 'S'
                total_seconds = hour * 3600 + phase_min * 60 + phase_sec
                phase_time = get_phase_utc(
                    year, month, day, hour, min, sec, phase_min, phase_sec
                )
                mag_picks_list.append(
                    [
                        station,
                        phase_time,
                        total_seconds,
                        phase_type,
                        dist,
                        azi,
                        toa,
                        elevation,
                        event_index,
                    ]
                )
        self.ymd = f'{year:4}{month:02}{day:02}'
        # Write it into dataframe
        self.df_h3dd_events = pd.DataFrame(mag_event_list, columns=mag_event_header)
        self.df_h3dd_picks = pd.DataFrame(mag_picks_list, columns=mag_picks_header)

    def _kinethreshold(
        self, comp_list: list, station: str, t1: UTCDateTime, t2: UTCDateTime
    ):
        raw_list = []
        detrend_list = []
        for comp in comp_list:
            sac_path = list(
                (self.sac_parent_dir / self.ymd).glob(f'*{station}*{comp}*')
            )[0]
            # 1. get the maxcN and mincN
            st = read(sac_path)
            st.trim(starttime=t1, endtime=t2)
            raw_list.extend([max(st[0].data), abs(min(st[0].data))])

            # 2. get the maxnN and minnN
            st.detrend('demean')
            st.detrend('linear')
            detrend_list.extend([max(st[0].data), abs(min(st[0].data))])
        seis_type = comp[1]
        max_amp = max(raw_list)
        max_detrend_amp = max(detrend_list)
        if seis_type == 'L':
            if max_amp <= 0.1:
                return 0
            else:
                return 1
        elif seis_type == 'S':
            if max_amp > 1:
                return 0
            elif max_amp < 0.0003:
                return 2
            else:
                return 1
        elif seis_type == 'H':
            if max_detrend_amp >= 5000000:
                return 0
            elif max_amp < 0.005:
                return 2
            else:
                return 1
        else:
            raise ValueError(f'seis_type is not correct: {comp}: {seis_type}')

    def _calculate_time_window(self, df_sta_picks: pd.DataFrame) -> Tuple[UTCDateTime, UTCDateTime, UTCDateTime, UTCDateTime]:  # noqa: UP006
        """
        calcuate the time window for each station.

        output
            1. t1: start time of the time window for actual analysis.
            2. t2: end time of the time window for actual analysis.
            3. tt1: start time of the time window for removing response.
            4. tt2: end time of the time window for removing response.

        """
        if len(df_sta_picks) == 2:
            # N4 mode == 1
            t1 = (
                UTCDateTime(
                    df_sta_picks[df_sta_picks['phase_type'] == 'P']['phase_time'].iloc[
                        0
                    ]
                )
                - 3
            )
            t2 = (
                t1
                + (
                    df_sta_picks[df_sta_picks['phase_type'] == 'S'][
                        'total_seconds'
                    ].iloc[0]
                    - df_sta_picks[df_sta_picks['phase_type'] == 'P'][
                        'total_seconds'
                    ].iloc[0]
                )
                * 2
                + 5
            )
            if t2 - t1 > 80:
                t2 = t1 + 80
        elif len(df_sta_picks) == 1:
            if df_sta_picks['phase_type'].iloc[0] == 'P':
                # N4 mode == 2
                if float(df_sta_picks['dist'].iloc[0]) > 250:
                    t1 = UTCDateTime(df_sta_picks['phase_time'].iloc[0]) - 3
                    t2 = t1 + 80
                elif float(df_sta_picks['dist'].iloc[0]) > 100:
                    t1 = UTCDateTime(df_sta_picks['phase_time'].iloc[0]) - 3
                    t2 = t1 + 40
                else:
                    t1 = UTCDateTime(df_sta_picks['phase_time'].iloc[0]) - 3
                    t2 = t1 + 20

                pass
            elif df_sta_picks['phase_type'].iloc[0] == 'S':
                # N4 mode == 3
                t1 = UTCDateTime(df_sta_picks['phase_time'].iloc[0]) - 3
                t2 = t1 + 30
        tt1 = t1 - 100
        tt2 = t2 + 100
        return t1, t2, tt1, tt2
    
    def _find_max_amp(self, code: int, comp_list: list, station: str, t1: UTCDateTime, t2: UTCDateTime, tt1: UTCDateTime, tt2: UTCDateTime, mn=1000, pre_filt=pre_filt, wa_simulate=wa_simulate):
        """
        Find maximum amplitude from different scenarios followed the
        CWA (1993).
        """
        if code == 0:
            logging.info(f'{station}: code == 0')
            return

        # for code == 1 & code == 2
        response_dict = {}
        for i, comp in enumerate(comp_list):
            sac_path = list(
                (self.sac_parent_dir / self.ymd).glob(f'*{station}*{comp}*')
            )[0]
            pz_path = list(self.pz_path.glob(f'*{station}*{comp}*'))[0]
            st = read(sac_path)
            attach_paz(tr=st[0], paz_file=str(pz_path))
            st.trim(starttime=tt1, endtime=tt2)  # cut longer for simulate
            st.simulate(
                paz_remove='self', paz_simulate=wa_simulate, pre_filt=pre_filt
            )
            if code == 2:
                st.filter('bandpass', freqmin=1, freqmax=25)
            st.trim(starttime=t1, endtime=t2)
            response_dict[i] = max(max(st[0].data), abs(min(st[0].data)))
        
        if comp[1] == 'L' or comp[1] == 'H':
            response_dict = {
                key: value * mn for key, value in response_dict.items()
            }

        return response_dict
    def _calulate_mag(self, response_dict: dict, dist: float, depth: float)-> float:
        """
        Using R (dist**2 + depth**2) to correct and estimate the magnitude.
        """
        nloga = math.log10(
            math.sqrt(response_dict[0] ** 2 + response_dict[1] ** 2)
        )

        R = math.sqrt(dist**2 + depth**2)
        if depth <= 35:
            if 0 < dist <= 80:
                dA = -0.00716 * R - math.log10(R) - 0.39
            elif dist > 80:
                dA = -0.00261 * R - 0.83 * math.log10(R) - 1.07
        else:
            dA = -0.00326 * R - 0.83 * math.log10(R) - 1.01

        return nloga - dA
    
    def get_mag(
        self, df_h3dd_events: pd.DataFrame, df_h3dd_picks: pd.DataFrame, lock, event_index: int
    ):
        """
        Calculate suitable time window for each station in N4 determination.
        """
        df_event = df_h3dd_events[df_h3dd_events['event_index'] == event_index].copy()
        depth = float(df_event['depth'].iloc[0])
        df_picks = df_h3dd_picks[(df_h3dd_picks['event_index'] == event_index)].copy()

        station_num = 0
        sum_mag = 0

        for station in set(df_picks['station']):
            df_sta_picks = df_picks[df_picks['station'] == station].copy()
            
            t1, t2, tt1, tt2 = self._calculate_time_window(df_sta_picks)
            
            sac_path_list = list((self.sac_parent_dir / self.ymd).glob(f'*{station}*'))
            comp_list = [
                sac_path.stem.split('.')[3]
                for sac_path in sac_path_list
                if sac_path.stem.split('.')[3][-1] != 'Z'
            ]
            if len(comp_list) == 2:  # ensuring we have 2 horizontal component.
                code = self._kinethreshold(
                    comp_list=comp_list, station=station, t1=t1, t2=t2
                )
                
                response_dict = self._find_max_amp(
                    code=code, comp_list=comp_list, station=station,
                    t1=t1, t2=t2, tt1=tt1, tt2=tt2
                )

                actual_depth = depth + df_sta_picks['elevation'].iloc[0] 
                if response_dict is not None:
                    sta_mag = self._calulate_mag(
                        response_dict=response_dict,
                        dist = float(df_sta_picks['dist'].iloc[0]),
                        depth=actual_depth
                        )
                    
                    station_num += 1
                    df_picks.loc[df_picks['station'] == station, 'magnitude'] = sta_mag
                    sum_mag += sta_mag
            else:
                logging.info(f"We only have {len(comp_list)} horizontal components in {station}")
                continue
        if station_num > 0:
            sum_mag /= station_num
            df_event['magnitude'] = sum_mag
        
        return df_event, df_picks
    
    def run_mag(self, output_dir: Path, processes=10):
        """
        Spawn processes to run `get_mag` for multiple event indices in parallel.
        """
        self.process_h3dd()
        
        event_indices = set(self.df_h3dd_events['event_index'])
        manager = mp.Manager()
        # shared_df_events = manager.dict()  # Store df_event for each event index.
        # shared_df_picks = manager.dict()  # Store df_picks for each event index.
        lock = manager.Lock()

        with mp.Pool(processes=processes) as pool:
            results = pool.starmap(
                self.get_mag,
                [(self.df_h3dd_events, self.df_h3dd_picks, lock, event_index) 
                for event_index in event_indices]
            )
        
        # Collect all events and picks into lists
        all_events, all_picks = [], []
        for event, picks in results:
            if not event.empty:
                all_events.append(event)
                df_event_result = pd.concat(all_events, ignore_index=True)
                df_event_result.to_csv(output_dir/"mag_events.csv", index=False)        

            if not picks.empty:
                all_picks.append(picks)
                df_picks_result = pd.concat(all_picks, ignore_index=True)
                df_picks_result.to_csv(output_dir/"mag_picks.csv", index=False)
