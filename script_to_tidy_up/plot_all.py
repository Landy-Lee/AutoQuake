#%%
import os
import glob
import math
import logging
import calendar
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import multiprocessing as mp
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from obspy import read, UTCDateTime
from obspy.imaging.beachball import beach
from geopy.distance import geodesic
from collections import defaultdict
from pyrocko import moment_tensor as pmt
from pyrocko.plot import beachball, mpl_color

def formatting(num):
    """Format single digit numbers with a leading zero."""
    return f"0{num}" if len(num) == 1 else num

def get_azi_toa_pol(dout_index_file, event_index):
    """Parse azimuth, takeoff angle, and polarity from the output file."""
    with open(dout_index_file, 'r') as r:
        lines = r.read().splitlines()
    beachball_dict = {}
    for line in lines:
        if line.strip()[:1].isalpha() and int(line.split()[-1]) == event_index: # if your station start with digit, modify the logic.
            sta = line[1:5].strip()
            azi = int(line[12:15].strip())
            toa = int(line[16:19].strip())
            polarity = line[19:20]
            beachball_dict[sta] = {'azimuth': azi, 'takeoff_angle': toa, 'polarity': polarity}
    return beachball_dict

def get_azi_toa_pol_v2(event_time, dout_file, analyze_year):
    with open(dout_file, 'r') as r:
        lines = r.readlines()
    beachball_dict = {}
    counter = 0
    passer = False
    for line in lines:
        parts = line.strip()
        if parts[:4] == analyze_year:
            year = parts[:4].strip()
            month = parts[4:6].strip()
            day = parts[6:8].strip()
            hh = parts[8:10].strip()
            mm = parts[10:12].strip()
            ss = parts[13:18].strip()
            compare_time = f"{year}{formatting(month)}{formatting(day)}T{hh}:{mm}:{ss}"
            if abs(UTCDateTime(event_time) - UTCDateTime(compare_time)) < 1:
                passer = True
                counter = 1
            else:
                counter = counter * 10 # once we find the event, counter = 10 not 0
                passer = False
                if counter != 0:
                    logging.info('we already find the match event, break')
                    break
        elif passer:
            sta = line[1:5].strip()
            azi = int(line[12:15].strip())
            toa = int(line[16:19].strip())
            polarity = line[19:20]
            beachball_dict[sta] = {'azimuth': azi, 'takeoff_angle': toa, 'polarity': polarity}
    return beachball_dict


def get_beach(source, focal_info, beachball_dict, color, ax):
    """Plot the beachball diagram."""
    mt = pmt.MomentTensor(strike=focal_info['plane'][0], dip=focal_info['plane'][1], rake=focal_info['plane'][2])
    ax.set_axis_off()
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.2, 1.2)
    projection = 'lambert'

    beachball.plot_beachball_mpl(
        mt, ax,
        position=(0., 0.),
        size=2.0,
        color_t=mpl_color(color),
        projection=projection,
        size_units='data')
    for sta in beachball_dict.keys():
        takeoff = beachball_dict[sta]['takeoff_angle']
        azi = beachball_dict[sta]['azimuth']
        polarity = beachball_dict[sta]['polarity']
        if polarity == ' ':
            continue
        # to spherical coordinates, r, theta, phi in radians
        # flip direction when takeoff is upward
        rtp = np.array([[
            1.0 if takeoff <= 90. else -1.,
            np.deg2rad(takeoff),
            np.deg2rad(90.-azi)]])
        # to 3D coordinates (x, y, z)
        points = beachball.numpy_rtp2xyz(rtp)

        # project to 2D with same projection as used in beachball
        x, y = beachball.project(points, projection=projection).T
        ax.plot(
            x, y,
            '+' if polarity == '+' else 'o',
            ms=10. if polarity == '+' else 10./np.sqrt(2.),
            mew=2.0,
            mec='black',
            mfc='none')
    ax.text(-0.9, -1.35, f"{source}\nQuality index: {focal_info['score']}\nStrike: {focal_info['plane'][0]}, Dip: {focal_info['plane'][1]}, Rake: {focal_info['plane'][2]}", fontsize=20)

def find_focal(event_time, focal):
    """Find focal mechanism information for a given event time."""
    with open(focal, 'r') as focal_list:
        focal_lines = focal_list.readlines()
    check_pass = False
    for line in focal_lines:
        parts = line.split()
        d = parts[0].replace('/','-')
        h = parts[1].split(':')[0]
        m = parts[1].split(':')[1]
        s = parts[1].split(':')[2]
        # avoiding time format problem
        if s == '60':
            logging.info(f"catch second foramt it in {line}")
            s = '00'
            m = formatting(str(int(m) + 1))
            if m == '60':
                logging.info(f"catch minute foramt it in {line}")
                m = '00'
                h = formatting(str(int(h)+1))
        if m == '60':
            logging.info(f"catch minute foramt it in {line}")
            m = '00'    
            h = formatting(str(int(h)+1))
        focal_utc = f'{d}T{h}:{m}:{s}'
        if abs(UTCDateTime(event_time)-UTCDateTime(focal_utc)) < 1:
            check_pass = True
            focal_info = defaultdict()
            strike = int(parts[6].split('+')[0])
            dip = int(parts[8].split('+')[0])
            rake = int(parts[10].split('+')[0])
            score = parts[12]
            focal_info = {'plane': [strike, dip, rake], 'score': score, 'status': True}
            break
    if not check_pass:
        focal_info = {'status': False}
    return focal_info

def find_mag(event_time, mag):
    """Find magnitude information for a given event time."""
    with open(mag, 'r') as mag_list:
        mag_lines = mag_list.readlines()
    check_pass = False
    for line in mag_lines:
        year = line[:4]
        month = line[4:6]
        day = line[6:8]
        hour = line[9:11]
        min = line[11:13]
        sec_int = line[13:15]
        sec_float = line[15:17]
        # avoiding time format problem
        if int(sec_int) >= 60:
            logging.info(f"catch second foramt it in {line} from find_mag")
            sec_int = formatting(str(int(sec_int) - 60))
            min = formatting(str(int(min) + 1))
            if int(min) >= 60:
                logging.info(f"catch minute foramt it in {line}")
                min = formatting(str(int(min) - 60))
                hour = formatting(str(int(hour)+1))
        if int(min) >= 60:
                logging.info(f"catch minute foramt it in {line}")
                min = formatting(str(int(min) - 60))
                hour = formatting(str(int(hour)+1))
        mag_utc = f'{year}-{month}-{day}T{hour}:{min}:{sec_int}.{sec_float}'
        if abs(UTCDateTime(event_time)-UTCDateTime(mag_utc)) < 1:
            check_pass = True
            mag_info = defaultdict()
            magnitude = round(float(line.split()[5]),2)
            mag_info = {'mag': magnitude, 'status': True}
            break
    if not check_pass:
        mag_info = {'status': False}
    return mag_info

def mag_annotate(event_time, gamma_mag, cwa_mag, ax, map_proj):
    """Annotate magnitude information on the map."""
    gamma_mag_info = find_mag(event_time, gamma_mag)
    cwa_mag_info = find_mag(event_time, cwa_mag)
    magnitude = 'NaN'
    # cwa ball
    i = -1
    if cwa_mag_info['status']:
        i += 1
        x, y = map_proj.transform_point(x=map_lon_max-0.7, y=map_lat_min+0.1*(i*2+1),
                                    src_crs=ccrs.Geodetic())
        ax.text(x, y, f"CWA Magnitude: {cwa_mag_info['mag']}", fontsize=15)
    # gamma ball
    if gamma_mag_info['status']:
        i += 1
        magnitude = round(gamma_mag_info['mag'])
        x, y = map_proj.transform_point(x=map_lon_max-0.7, y=map_lat_min+0.1*(i*2+1),
                                    src_crs=ccrs.Geodetic())
        ax.text(x, y, f"N4 Magnitude: {gamma_mag_info['mag']}", fontsize=15)
    return magnitude

def beach_plot(event_time, gamma_focal, cwa_focal, ax, map_proj):
    gamma_focal_info = find_focal(event_time, gamma_focal) # we might modify this but not now.
    cwa_focal_info = find_focal(event_time, cwa_focal)
    gamma_check = False
    cwa_check = False
    # gamma ball
    i = -1
    if gamma_focal_info['status']:
        gamma_check = True
        i += 1
        x, y = map_proj.transform_point(x=map_lon_min+0.2, y=map_lat_max-0.2*(i*2+1),
                                    src_crs=ccrs.Geodetic())
        b= beach(gamma_focal_info['plane'], xy=(x, y), width=0.3, linewidth=1, facecolor='b')
        ax.add_collection(b)
        ax.text(x+0.2, y-0.1, f"GaMMA\nQuality index: {gamma_focal_info['score']}\nStrike: {gamma_focal_info['plane'][0]}, Dip: {gamma_focal_info['plane'][1]}, Rake: {gamma_focal_info['plane'][2]}", fontsize=15)
    # cwa ball
    if cwa_focal_info['status']:
        cwa_check = True
        i += 1
        x, y = map_proj.transform_point(x=map_lon_min+0.2, y=map_lat_max-0.2*(i*2+1),
                                    src_crs=ccrs.Geodetic())
        b= beach(cwa_focal_info['plane'], xy=(x, y), width=0.3, linewidth=1, facecolor='r')
        ax.add_collection(b)
        ax.text(x+0.2, y-0.1, f"CWA\nQuality index: {cwa_focal_info['score']}\nStrike: {cwa_focal_info['plane'][0]}, Dip: {cwa_focal_info['plane'][1]}, Rake: {cwa_focal_info['plane'][2]}", fontsize=15)

    status = 'no' # default status
    if gamma_check:
        status = 'gamma'
    if cwa_check:
        status = 'cwa'
    if gamma_check and cwa_check:
        status = 'both'
    return status

def cat_files(directory, pattern, num_files, output_filename):
    """Concatenate files matching the pattern into one file."""
    directory_path = Path(directory)
    output_file = directory_path / output_filename

    contents = []

    for i in range(num_files):
        file_path = directory_path / pattern.format(i)
        with file_path.open('r') as file:
            contents.append(file.read())

    with output_file.open('w') as output:
        for content in contents:
            output.write(content)
    return directory_path / output_filename

def date_range(start_date, end_date):
    """Generate a list of dates between start_date and end_date."""
    start_date = datetime.strptime(str(start_date), "%Y%m%d")
    end_date = datetime.strptime(str(end_date), "%Y%m%d")
    delta = end_date - start_date
    return [(start_date + timedelta(days=i)).strftime("%Y%m%d") for i in range(delta.days + 1)]


# Read data
def convert(date):
    logging.basicConfig(filename=os.path.join(convert_parent_dir, 'convert.log'), level=logging.INFO, filemode='w')
    csv_path = os.path.join(picks_result_dir, date, 'picks.csv')
    df = pd.read_csv(csv_path)

    # Loop through DataFrame rows
    # underline is the conventional way in pandas to discard the item that is no needed.
    # in this case, iterrows will return (idx, series), and series is the crucial part we need.
    for _, row in df.iterrows():
        phase_time = row['phase_time']
        file_name = row['file_name']
        phase_type = row['phase_type']
        phase_score = row['phase_score']

        # Create date directory
        time_split = phase_time.split('T')
        ymd = time_split[0].replace('-', '')
        ymd_dir = os.path.join(convert_parent_dir, ymd)
        os.makedirs(ymd_dir, exist_ok=True)

        # Create phase type directory
        phase_type_dir = os.path.join(ymd_dir, phase_type)
        os.makedirs(phase_type_dir, exist_ok=True)

        # Extract station name
        station = file_name.split('.')[1]

        # Convert time to seconds
        time_str = time_split[1]
        time_obj = datetime.strptime(time_str, '%H:%M:%S.%f')
        total_seconds = time_obj.second + time_obj.minute * 60 + time_obj.hour * 3600 + time_obj.microsecond / 1000000.0

        # Write data to file
        output_file = os.path.join(phase_type_dir, f"{station}_{phase_type}.dat")
        with open(output_file, 'a') as f:
            f.write(f"{total_seconds} {phase_score}\n")
            logging.info(f'output_file write down a new data!')

def add_event_index(target_file, output_file, analyze_year):
    
    # because we use the append mode for writing data, remove the existing file before writing is needed.
    if os.path.exists(output_file):
        os.remove(output_file)
    
    event_index = 0
    with open(target_file, 'r') as r:
        lines = r.readlines()
        for line in lines:
            if line.strip()[:4] == analyze_year: # the event list start with digit (e.g. '2'0240401)
                event_index += 1
                with open(output_file, 'a') as eve:
                    eve.write(f"{' ':1}{line.strip()}{' ':1}{event_index:<5}\n")
            else: 
                with open(output_file, 'a') as pick:
                    pick.write(f"{' ':1}{line.strip()}{' ':1}{event_index:<5}\n")

    return output_file

def header_count(header_file):
    """Count the number of lines in the header file."""
    with open(header_file, 'r') as r:
        return len(r.read().splitlines())

def to_seconds_datetime(date, hour, minute, sec):
    y = date[:4]
    m = date[4:6]
    d = date[6:8]
    # Convert the seconds to an integer
    hh = int(hour.strip())
    mm = int(minute.strip())
    ss = float(sec.strip())
    total_seconds = hh*3600 + mm*60 + ss
    # Format the output string
    utc_datetime = f"{y}-{m}-{d}T{hh:02d}:{mm:02d}:{ss:05.2f}"
    return utc_datetime, total_seconds

def degree_trans(part):
    """Transform degree-minute-second to decimal degrees."""
    if len(part) == 7:
        deg = int(part[:2])
        dig = int(part[2:4]) / 60 + int(part[5:]) / 3600
    else:
        deg = int(part[:3])
        dig = int(part[3:5]) / 60 + int(part[6:]) / 3600
    return deg+dig

def sta_associated(filename, event_i):
    """Extract station association data from the file."""
    with open(filename,'r') as r:
        lines = r.readlines()
    station_associated = defaultdict(dict)
    station_aso_list = []
    for line in lines:
        #first_char = line.strip()[0] # first character of the line, should we standardize the naming idiom of station? 
        if line.strip()[:4] == analyze_year and int(line.strip().split()[-1]) == event_i: 
            date = line.strip()[:8]
            hour = line.strip()[8:10]
            minute = line.strip()[10:12]
            sec = line.strip()[12:18]
            event_part = line.strip().split()
            event_time, event_total_seconds = to_seconds_datetime(date, hour, minute, sec)
            lat_part = line.strip()[18:25]
            lon_part = line.strip()[25:33]
            event_lon = round(degree_trans(lon_part),2)
            event_lat = round(degree_trans(lat_part),2)
            event_depth = float(event_part[-2])
            evt_point = (event_lat, event_lon)
        elif int(line.strip().split()[-1]) == event_i:
            pick_part = line.strip().split()
            station_name = pick_part[0]
            if station_name not in station_aso_list:
                station_aso_list.append(station_name)
            pick_min = pick_part[4]
            if pick_part[7] == '1.00':
                pick_type = 'P'
                pick_sec = pick_part[5]
                # Situation A: minute > 60 need to plus 1 to the hours
                if minute == '59' and pick_min == '60':
                    pick_min = 0
                    hour = int(hour) + 1
                    # Situation A*: not only the minute > 60, the hour also = 24, which means a new day, day needs to plus 1
                    if hour == 24:
                        hour = 0
                        #day = int(date[-2:]) + 1 
            else:
                pick_type = 'S'
                pick_sec = pick_part[8]
                # Situation A: minute > 60 need to plus 1 to the hours
                if minute == '59' and pick_min == '60':
                    pick_min = 0
                    hour = int(hour) + 1
                    # Situation A*: not only the minute > 60, the hour also = 24, which means a new day, day needs to plus 1
                    if hour == 24:
                        hour = 0
                        #day = int(date[-2:]) + 1
            pick_times = int(hour)*3600 + int(pick_min)*60 + float(pick_sec)
            station_associated[f"{station_name}_{pick_type}"] = {'pick_time': pick_times, 'type': pick_type}
    return date, station_associated, station_aso_list, event_total_seconds, event_time, event_lon, event_lat, event_depth, evt_point

def sta_data(event_time, station_path, sac_path_parent_dir, sac_dir_name, evt_point):
    """Retrieve station data."""
    starttime_trim = UTCDateTime(event_time) - 30
    endtime_trim = UTCDateTime(event_time) + 60
    df = pd.read_csv(station_path)
    station_list = df['station'].to_list() # convert the dtype from object to list
    # plotting all the waveform
    station_data = defaultdict(dict)
    date = ''.join(event_time.split('T')[0].split('-')) # this would like split 2024-04-01 into 20240401, match the naming paradigm of ours.
    sac_path = os.path.join(sac_path_parent_dir, date, sac_dir_name)
    for s in station_list:
        # glob the waveform
        data_path_glob = os.path.join(sac_path, f"*{s}.*Z.*")
        data_path = glob.glob(data_path_glob)
        sta_all_lon = df[df['station'] == s]['lon'].iloc[0]
        sta_all_lat = df[df['station'] == s]['lat'].iloc[0]
        sta_point = (sta_all_lat, sta_all_lon)
        # read the waveform
        try:
            st = read(data_path[0])
            st.taper(type='hann', max_percentage=0.05)
            st.filter("bandpass", freqmin=1, freqmax=10)
            dist = geodesic(evt_point, sta_point).km
            dist_round = np.round(dist ,1)
            st_check = 0
            if starttime_trim < st[0].stats.starttime:
                st_check = 1
            else:
                st_check = 0
            st[0].trim(starttime=starttime_trim, endtime=endtime_trim)
            sampling_rate = 1 / st[0].stats.sampling_rate
            time_sac = np.arange(0, 90+sampling_rate,sampling_rate) # using array to ensure the time length as same as time_window.
            x_len = len(time_sac)
            data_sac_raw = st[0].data / max(st[0].data) # normalize the amplitude.
            data_sac_raw = data_sac_raw*amplify_index + dist # amplify the amplitude. Here we multiply to 100.
            # we might have the data lack in the beginning:
            if st_check == 1:
                data_sac = np.pad(data_sac_raw, (x_len - len(data_sac_raw), 0), mode='constant', constant_values=np.nan) # adding the Nan to ensure the data length as same as time window.
            else:    
                data_sac = np.pad(data_sac_raw, (0, x_len - len(data_sac_raw)), mode='constant', constant_values=np.nan) # adding the Nan to ensure the data length as same as time window.
            station_data[str(s)] = {'time':time_sac, 'sac_data': data_sac, 'distance': dist_round}
        except Exception as e:
            logging.info(f"Error: {e} existed, we don't have {s} data around {event_time}")
            continue
    return station_data

def all_pick_time(date, event_total_seconds):
    """Retrieve pick times for all stations."""
    types = ['P','S']
    all_picks_dict = defaultdict(dict)
    for type in types:
        type_dir = os.path.join(convert_parent_dir, date, type)
        type_picks = os.listdir(type_dir)
        for pick in type_picks:
            time_box = []
            with open(os.path.join(type_dir, pick), 'r') as r:
                lines = r.readlines()
                for line in lines:
                    desired_time = float(line.split()[0])
                    if desired_time >= event_total_seconds -30 and desired_time <= event_total_seconds + 60:
                        time_box.append(line.split()[0])
                    else:
                        continue
            key_name = pick.split('.')[0]
            if time_box: #same as if time_box != []
                all_picks_dict[key_name] = {'pick_time':time_box, 'type':type}
    return all_picks_dict

def scatter_dict_gen(all_picks_dict, station_data, event_total_seconds):
    """Generate scatter plot data."""
    scatter_dict = defaultdict(dict)
    for key in list(all_picks_dict.keys()):
        times = all_picks_dict[key]['pick_time']
        type = all_picks_dict[key]['type']
        station = key.split('_')[0]
        x_box = [float(t) - (event_total_seconds - 30) for t in times]
        y_box = [station_data[station]['distance']] * len(times)
        scatter_dict[key] = {'x':x_box, 'y':y_box, 'type':type}
    return scatter_dict

def interval_judge(map_min, map_max):
    """Determine major and minor intervals for ticks."""
    if abs(map_max - map_min) <= 1.5:
        major_interval = 0.5
        minor_interval = 0.1
    else:
        major_interval = 1
        minor_interval = 0.5
    return major_interval, minor_interval

def focal_sta_time(dout_index_file, event_i, analyze_year):
    """Retrieve focal station times from the output file."""
    with open(dout_index_file,'r') as r:
        lines = r.readlines()
    sta_time_dict = {}
    for line in lines:
        parts = line.strip()
        if int(parts.split()[-1]) == event_i:
            if parts[:4] == analyze_year:
                year = parts[:4].strip()
                month = parts[4:6].strip()
                day = parts[6:8].strip()
                hh = parts[8:10].strip()
                mm = parts[10:12].strip()
                ss = parts[13:18].strip()
                ymd = f"{year}{formatting(month)}{formatting(day)}"
                timestamp = f"{year}{formatting(month)}{formatting(day)}T{hh}:{mm}:{ss}"
            else:
                sta_name = parts[:4].strip()
                p_mm = parts[20:22].strip()
                p_ss = parts[23:28].strip()
                polarity = parts[18:19]
                if mm == '59' and p_mm == '60':
                    format_mm = 0
                    format_hh = int(hh) + 1
                    # Situation A*: not only the minute > 60, the hour also = 24, which means a new day, day needs to plus 1
                    if format_hh == 24:
                        new_format_hh = 0
                        format_day = int(day) + 1
                        # Situation A**: if days > monthrange, month need to plus 1
                        if calendar.monthrange(int(year), int(month))[1] < format_day:
                            format_month = int(month) + 1
                            new_format_day = 1   
                            p_arrival = UTCDateTime(int(year), format_month, new_format_day, new_format_hh, format_mm, float(p_ss))
                        else:
                            p_arrival = UTCDateTime(int(year), int(month), format_day, new_format_hh, format_mm, float(p_ss))
                    else:
                        p_arrival = UTCDateTime(int(year), int(month), int(day), format_hh, format_mm, float(p_ss))
                else:
                    p_arrival = UTCDateTime(int(year), int(month), int(day), int(hh), int(p_mm), float(p_ss))
                sta_time_dict[sta_name] = {'pick_time': p_arrival, 'polarity': polarity}
    return sta_time_dict, ymd, timestamp

def focal_sta_data(sta_time_dict, ymd, data):
    """Retrieve focal station data."""
    station_list = list(sta_time_dict.keys())
    # plotting all the waveform
    station_data = defaultdict(dict)
    sac_path = os.path.join(data, ymd, 'data_final')
    for s in station_list:
        # trim time
        arrival_time = sta_time_dict[s]['pick_time']
        train_starttime_trim = arrival_time - train_window
        train_endtime_trim = arrival_time + train_window
        window_starttime_trim = arrival_time - visual_window
        window_endtime_trim = arrival_time + visual_window
        # glob the waveform
        data_path_glob = os.path.join(sac_path, f"*{s}.*Z.*")
        data_path = glob.glob(data_path_glob)
        # read the waveform
        try:
            st = read(data_path[0])
            st.detrend('demean')
            st.detrend('linear')
            st.filter("bandpass", freqmin=1, freqmax=10)
            st.taper(0.001)
            st.resample(sampling_rate=100)
            # this is for visualize length
            st[0].trim(starttime=window_starttime_trim, endtime=window_endtime_trim)
            visual_time = np.arange(0, 2*visual_window+1/sampling_rate, 1/sampling_rate) # using array to ensure the time length as same as time_window.
            visual_sac = st[0].data 
            # this is actual training length
            st[0].trim(starttime=train_starttime_trim, endtime=train_endtime_trim)
            train_time = np.arange(1.36, 1.36 + 2*train_window+1/sampling_rate, 1/sampling_rate) # using array to ensure the time length as same as time_window.
            train_sac = st[0].data 
            station_data[s] = {'visual_time':visual_time, 'visual_sac': visual_sac, 'train_time':train_time, 'train_sac': train_sac}
            
        except Exception as e:
            logging.info(f"Error: {e} existed, we don't have {s} data in {ymd}")
    return station_data

def plot_assoc(i):
    """Main function to plot association data for a given event index."""
    logging.info(f"This is event_{i}")

    date, station_associated, station_aso_list, event_total_seconds, event_time, event_lon, event_lat, event_depth, evt_point = sta_associated(filename=dat_output_file, event_i=i)

    station_data = sta_data(event_time=event_time,station_path=station_info_path, sac_path_parent_dir=sac_path_parent_dir, sac_dir_name = sac_dir_name, evt_point=evt_point)

    all_picks_dict = all_pick_time(date=date, event_total_seconds=event_total_seconds)

    scatter_dict = scatter_dict_gen(all_picks_dict=all_picks_dict, station_data=station_data, event_total_seconds=event_total_seconds)

    # plotting
    map_proj = ccrs.PlateCarree() # declare first to pass the if condition below.
    tick_proj = ccrs.PlateCarree()

    # Scaling the figure

    # First, ensuring the focal plane is available
    gamma_focal_info = find_focal(event_time, gamma_focal)
    cwa_focal_info = find_focal(event_time, cwa_focal)
    logging.info(cwa_focal_info)
    if gamma_focal_info['status'] and cwa_focal_info['status']:
        fig = plt.figure(figsize=(36,20))
        gs1 = GridSpec(10, 18, left=0, right=0.9, top=0.95 , bottom=0.42 , wspace=0.1)
        ax1 = fig.add_subplot(gs1[1:, :5], projection=map_proj) # cartopy axes
        ax2 = fig.add_subplot(gs1[:, 9:18]) # arrival-time axes (although it being called as ax2, it's the rightmost figure!)
        n_cols = 5
        ax3 = fig.add_subplot(gs1[:5, 5:9]) # focal axes-upper part
        ax4 = fig.add_subplot(gs1[5:, 5:9]) # focal axes-lower part

        # gamma ball
        # Prepare the dict contains the azimuth, takeoff angle, polarity
        gamma_beachball_dict = get_azi_toa_pol(gamma_dout_index_file, i)
        get_beach('GaMMA',gamma_focal_info, gamma_beachball_dict, color='red', ax=ax3)

        # cwa ball
        cwa_beachball_dict = get_azi_toa_pol_v2(event_time, cwa_dout_file, analyze_year)
        logging.info(cwa_beachball_dict)
        get_beach('CWA',cwa_focal_info, cwa_beachball_dict, color='skyblue2', ax=ax4)

        focal_status = 'both'

    elif gamma_focal_info['status'] or cwa_focal_info['status']:
        fig = plt.figure(figsize=(36,20))
        gs1 = GridSpec(10, 18, left=0, right=0.9, top=0.95 , bottom=0.42 , wspace=0.1)
        ax1 = fig.add_subplot(gs1[1:, :5], projection=map_proj) # cartopy axes
        ax2 = fig.add_subplot(gs1[:, 9:18]) # arrival-time axes
        n_cols = 5
        ax3 = fig.add_subplot(gs1[:5, 5:9]) # focal axes

        if gamma_focal_info['status']:
            gamma_beachball_dict = get_azi_toa_pol(gamma_dout_index_file, i)
            get_beach('GaMMA',gamma_focal_info, gamma_beachball_dict, color='red', ax=ax3)
            focal_status = 'gamma'
            
        if cwa_focal_info['status']:
            cwa_beachball_dict = get_azi_toa_pol_v2(event_time, cwa_dout_file, analyze_year)
            get_beach('CWA',cwa_focal_info, cwa_beachball_dict, color='skyblue2', ax=ax3)
            focal_status = 'cwa'
    else:
        fig = plt.figure(figsize=(24,20))
        gs1 = GridSpec(10, 10, left=0, right=0.9, top=0.95 , bottom=0.42 , wspace=0.1)
        ax1 = fig.add_subplot(gs1[1:, :4], projection=map_proj)
        ax2 = fig.add_subplot(gs1[:, 4:])
        n_cols = 4
        focal_status = 'no' # default status

    ## cartopy
    #f_name = "/home/patrick/.local/share/cartopy/shapefiles/natural_earth/Raster/NE1_HR_LC_SR_W.tif"
    region = [map_lon_min, map_lon_max , map_lat_min , map_lat_max]
    #ax1.imshow(plt.imread(f_name), origin='upper', transform=map_proj, extent=[-180, 180, -90, 90])
    ax1.coastlines()
    ax1.set_extent(region)
    ax1.add_feature(cfeature.LAND)
    ax1.add_feature(cfeature.OCEAN)
    ax1.add_feature(cfeature.COASTLINE)
    df = pd.read_csv(station_info_path)
    # plotting epicenter
    ax1.scatter(x=event_lon, y=event_lat, marker="*", color='gold', s=400, zorder=4)
    # plotting all stations
    ax1.scatter(x=df['lon'], y=df['lat'],marker="^", color='silver', s=300, zorder=2)
    # plotting associated stations
    for station in df['station']:
        if station in station_aso_list:
            ax1.scatter(x=df[df['station'] == station]['lon'], y=df[df['station'] == station]['lat'],marker="^", color='r', s=300, zorder=3)
    # plot focal
    # focal_status = beach_plot(event_time, gamma_focal, cwa_focal, ax1, map_proj)
    round_mag = mag_annotate(event_time, gamma_mag, cwa_mag, ax1, map_proj) # annotating the magnitude
    # map setup
    ## interval judge
    lon_major_interval, lon_minor_interval = interval_judge(map_lon_min, map_lon_max)
    lat_major_interval, lat_minor_interval = interval_judge(map_lat_min, map_lat_max)

    ## ticks boundary & format
    ax1.set_xticks(np.arange(map_lon_min, map_lon_max, lon_major_interval), crs=tick_proj)
    ax1.set_xticks(np.arange(map_lon_min, map_lon_max, lon_minor_interval), minor=True, crs=tick_proj)
    ax1.set_yticks(np.arange(map_lat_min, map_lat_max, lat_major_interval), crs=tick_proj)
    ax1.set_yticks(np.arange(map_lat_min, map_lat_max, lat_minor_interval), minor=True, crs=tick_proj)
    ax1.xaxis.set_major_formatter(LongitudeFormatter())
    ax1.yaxis.set_major_formatter(LatitudeFormatter())
    ax1.tick_params(axis='both', which='major', length=3, width=1, labelsize=5)

    # setting title
    ax1.set_title(f"{event_time} \nLon:{event_lon}, Lat:{event_lat}, Depth:{event_depth}", fontsize = 30, pad=25)
    ax1.set_aspect('auto')

    ## waveform
    for sta in list(station_data.keys()):
        if sta not in station_aso_list:
            # plot the waveform transparent
            ax2.plot(station_data[sta]['time'], station_data[sta]['sac_data'], color='k', linewidth=0.4, alpha=0.25, zorder =1)
            ax2.text(station_data[sta]['time'][-1]+1, station_data[sta]['distance'], sta, fontsize=10, verticalalignment='center', alpha=0.25)
        else:
            ax2.plot(station_data[sta]['time'], station_data[sta]['sac_data'], color='k', linewidth=0.4, zorder =1)
            ax2.text(station_data[sta]['time'][-1]+1, station_data[sta]['distance'], sta, fontsize=10, verticalalignment='center')
            pass

    ## pick
    for sta_type in list(scatter_dict.keys()):
        x = scatter_dict[sta_type]['x']
        y = scatter_dict[sta_type]['y']
        if scatter_dict[sta_type]['type'] == 'P':
            ax2.scatter(x, y, color='r', s =3, zorder =2)
        else:
            ax2.scatter(x, y, color='c', s =3, zorder =2)
    for sta_type in list(station_associated.keys()):
        if not scatter_dict[sta_type]: # the dict is empty
            sta = sta_type.split('_')[0]
            tp = sta_type.split('_')[1]
            if tp == 'P':
                ax2.text(station_data[sta]['time'][-1]+1, station_data[sta]['distance'], sta_type, color='r', fontsize=4, verticalalignment='center')
            else:
                ax2.text(station_data[sta]['time'][-1]+1, station_data[sta]['distance'], sta_type, color='c', fontsize=4, verticalalignment='center')
        else:
            x = float(station_associated[sta_type]['pick_time']) - (event_total_seconds-30)
            y = scatter_dict[sta_type]['y'][0]
            if scatter_dict[sta_type]['type'] == 'P':
                ax2.plot([x, x], [y-bar_length, y+bar_length], color='r', linewidth=0.7, zorder =2) # associated_picks length setting
            elif scatter_dict[sta_type]['type'] == 'S':
                ax2.plot([x, x], [y-bar_length, y+bar_length], color='c', linewidth=0.7, zorder =2) # associated_picks length setting
            else:
                continue

    ## setting
    ax2.set_xlim(0,90)
    ax2.xaxis.set_minor_locator(MultipleLocator(5))
    ax2.yaxis.set_minor_locator(MultipleLocator(10))
    ax2.tick_params(axis='both', which='major', length=3, width=1, labelsize=10)
    ax2.set_yticklabels([]) # Remove the tick labels but keep the ticks
    #ax2.tick_params(axis='x', which='minor', length=4, width=0.5)
    ax2.set_xlabel("Time (s)", fontsize = 20)

    # check_focal
    focal_sta_time_dict, ymd, timestamp = focal_sta_time(gamma_dout_index_file, i, analyze_year)
    if focal_sta_time_dict: # pass empty dict
        focal_station_data = focal_sta_data(sta_time_dict=focal_sta_time_dict, ymd=ymd, data=sac_path_parent_dir)
        focal_station_list = list(focal_sta_time_dict.keys())
        wavenum = len(focal_station_list)
        n_rows = math.ceil(wavenum / n_cols)
        gs2 = GridSpec(n_rows, n_cols, left=0, right=0.9, top=0.37, bottom=0, hspace=0.4, wspace=0.05)
        for index, station in enumerate(focal_station_list):
            polarity = focal_sta_time_dict[station]['polarity']
            ii = index // n_cols
            jj = index % n_cols
            ax = fig.add_subplot(gs2[ii, jj])
            x_wide = focal_station_data[station]['visual_time']
            y_wide = focal_station_data[station]['visual_sac']
            ax.plot(x_wide, y_wide, color='k')
            ax.set_xlim(0, visual_window*2)
            ax.grid(True, alpha=0.7)
            x = focal_station_data[station]['train_time']
            y = focal_station_data[station]['train_sac']
            ax.plot(x,y, color='r')
            ax.set_title(f"{station}({polarity})", fontsize = 15)
            ax.set_xticklabels([]) # Remove the tick labels but keep the ticks
            ax.set_yticklabels([]) # Remove the tick labels but keep the ticks
            ax.xaxis.set_major_locator(MultipleLocator(0.5))
            ax.xaxis.set_minor_locator(MultipleLocator(0.1))
            ax.scatter(x[int(train_window*100)],y[int(train_window*100)], color = 'c', marker = 'o')


    filename = f"{event_time.replace(':','_').replace('.','_')}_focal_{focal_status}_mag_{round_mag}.png"
    figure_dir = os.path.join(parent_dir, date, 'Figure')
    os.makedirs(figure_dir, exist_ok=True)
    file_path = os.path.join(figure_dir, filename)
    plt.subplots_adjust(wspace=0.01)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f'event_{i} plotting over')  

if __name__ == '__main__':
    
    analyze_year = '2024' #string
    # Path
    parent_dir = '/home/patrick/Work/AutoQuake-repo/tests/data/for_plot_all/'
    station_info_path = '/home/patrick/Work/AutoQuake-repo/tests/data/for_plot_all/station.csv' # format: {station} {lon} {lat}
    gamma_focal = '/home/patrick/Work/playground/gamma_gafocal_20240401_20240417_results.txt'
    cwa_focal = '/home/patrick/Work/playground/cwb_gafocal_20240401_20240417_results.txt'
    gamma_mag = '/home/patrick/Work/playground/gamma_mag_catalog_20240401_20240417.txt'
    cwa_mag = '/home/patrick/Work/playground/cwb_mag_catalog_20240401_20240417.txt'
    gamma_dout_file = '/home/patrick/Work/AutoQuake-repo/tests/data/for_plot_all/gamma_all.dout' # checking focal
    cwa_dout_file = '/home/patrick/Work/AutoQuake-repo/tests/data/for_plot_all/cwa_all.dout'

    logging.basicConfig(filename=os.path.join(parent_dir, f'plot.log'),format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO, filemode='w')
    print('Go check the New loggerrrrrrrrrrrrrr')
    # simple lever
    # convert_check = True # check whether we have converted data or not. If False, then converting will be carry out.

    # if convert_check:
    convert_parent_dir = '/home/patrick/Work/AutoQuake/Hualien0403/phasenet_no_amplitude/20240401_20240417'
    # else:
    #     # we only needs date_list when we do not convert the format firstly
    #     start_date = 20220701 #int 
    #     end_date = 20220702 #int
    #     date_list = date_range(start_date, end_date)

    #     # Convert phasenet picks into assoloc form, which each station_phase (MACB_P) is a file, containing the picks time. (this part should be optimized)
    #     convert_parent_dir = os.path.join(parent_dir, "trans")
    #     picks_result_dir = os.path.join(parent_dir, "phasenet_picks") 
    #     os.makedirs(convert_parent_dir, exist_ok=True)
    #     with mp.Pool(processes=10) as pool:
    #         pool.map(convert, date_list) 

    # Counting the num of files and using cat_files() to concatenate.
    num_files = len(glob.glob(os.path.join(parent_dir, f"*gamma_new*")))
    target_file = cat_files(parent_dir, 'gamma_new_{}.dat_ch', num_files, 'gamma_all.dat_ch') 

    # adding the event index for iterating parallel
    dat_output_file = os.path.join(parent_dir, os.path.basename(target_file).replace('.dat_ch','.dat_idx')) 
    gamma_dout_index_file = os.path.join(parent_dir, os.path.basename(gamma_dout_file).replace('.dout','.dout_idx'))
    dat_output_file = add_event_index(target_file, dat_output_file, analyze_year) # for plotting 90 s waveform
    gamma_dout_index_file = add_event_index(gamma_dout_file, gamma_dout_index_file, analyze_year) # for plotting 1.64 s waveform + polarity

    # header file to count the iteration range.
    header_file = os.path.join(parent_dir, 'gamma_events_order.csv')
    event_num = header_count(header_file)
    event_num_list = range(1, event_num+1)
    # event_num_list = range(1, 7) # or denote by yourself

    # the parent directory of sac data
    sac_path_parent_dir = '/home/patrick/Work/AutoQuake/Hualien0403/20240401_20240428/Dataset' 
    sac_dir_name = 'data_final' # name of sac_dir, see the example directory

    # Visualizing para
    bar_length = 1 # length of associated bar
    amplify_index = 5 # modify this to adjuct the amplitude of waveform

    # Cartopy map setting
    map_lon_min = 119.7 # 119.7 for whole Taiwan
    map_lon_max = 122.5 # 122.5 for whole Taiwan
    map_lat_min = 21.7 # 21.7 for whole Taiwan
    map_lat_max = 25.4 # 25.4 for whole Taiwan
    
    # focal check waveform setting
    train_window = 0.64
    visual_window = 2
    sampling_rate = 100
    
    # mp
    with mp.Pool(processes=30) as pool:
        pool.map(plot_assoc, event_num_list) 
# %%
