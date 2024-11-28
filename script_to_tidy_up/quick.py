# %% Convert polarity.picks into h3dd finshed format.
from pathlib import Path

import pandas as pd

dout_path = Path(
    '/home/patrick/Work/AutoQuake_pamicoding/Reloc2/seis_das_new_0924_261.dat_ch.dout'
)
# %%
output_path = '/home/patrick/Work/EQNet/tests/hualien_0403/gamma_test/test_5/for_gafocal_seis_das_new_261.dout'
polarity_picks = '/home/patrick/Work/EQNet/tests/hualien_0403/gamma_test/test_5/gamma_events_polarity.csv'
df_pol = pd.read_csv(polarity_picks)

with open(output_path, 'w') as fo:
    with open(dout_path) as r:
        lines = r.readlines()
    counter = -1
    for line in lines:
        if line[1:5] == '2024':
            counter += 1
            print(counter)
            if counter == 35:
                continue
            fo.write(f'{line}')
        elif line[35:39] == '1.00':
            if counter == 35:
                continue
            station = line[:5].strip()
            df_line = df_pol[
                (df_pol['event_index'] == counter) & (df_pol['station_id'] == station)
            ]
            polarity = df_line.iloc[0].polarity
            if polarity == 'U':
                polarity = '+'
            elif polarity == 'D':
                polarity = '-'
            else:
                polarity = ' '
            first_part = line[:19]
            seconds_part = line[21:].strip()
            fo.write(
                f'{first_part}{polarity} {seconds_part} 0.00 0.00 0.00 0.00 0   0.0\n'
            )
# %% ========== Retrieve SAC data ==========

from pathlib import Path

from obspy import UTCDateTime, read

data_single = Path('/home/patrick/Work/Hualien0403/Dataset/20240402/data_single')
data_all = Path('/home/patrick/Work/Hualien0403/Dataset/20240402/data')
main_eq_dir = Path('/home/patrick/Work/Hualien0403/main_eq/dataset')
main_eq_dir.mkdir(parents=True, exist_ok=True)
main_time = UTCDateTime('2024-04-02T23:58:10')
file_list = Path('/home/patrick/Work/Hualien0403/main_eq/file_list')
# %%
with open(file_list, 'w') as f:
    for i in list(data_single.glob('*.SAC')):
        if i.name.split('.')[1][:2] == 'SM':
            continue
        st = read(i)
        starttime = UTCDateTime(st[0].stats.starttime)
        endtime = UTCDateTime(st[0].stats.endtime)
        # print(f"station: {i.name}, starttime: {starttime}, endtime: {endtime}")
        if starttime < main_time and endtime > main_time:
            st.write(f'{str(main_eq_dir)}/{i.name}', format='SAC')
            f.write(f'{str(i)}\n')

    for i in list(data_all.glob('*SM*HL*.SAC')):
        st = read(i)
        starttime = UTCDateTime(st[0].stats.starttime)
        endtime = UTCDateTime(st[0].stats.endtime)
        if starttime < main_time and endtime > main_time:
            st.write(f'{str(main_eq_dir)}/{i.name}', format='SAC')
            f.write(f'{str(i)}\n')
# %% ===== Response test =====
from obspy import UTCDateTime, read
from obspy.io.sac.sacpz import attach_paz

## velocity
pz_file = '/data/share/for_patrick/PZ_test/SAC_PZs_TW_SHUL_HH2_00_2018.121.00.00.00.0000_2599.365.23.59.59.99999'
prefilt = (0.1, 0.5, 30, 35)
starttime = UTCDateTime('2024-04-03T03:56:19')  # 14179
endtime = UTCDateTime('2024-04-03T03:58:19')  # 14299
# using attach_paz
st = read(
    '/home/patrick/Work/Hualien0403/Dataset/20240403/data_final/TW.SHUL.00.HH2.D.2024,094,00:27:51.SAC'
)
st.trim(starttime, endtime)
# %%
attach_paz(tr=st[0], paz_file=pz_file, tovel=True)
st.simulate(paz_remove='self', pre_filt=prefilt)
# using hand written dict
st_hand = read(
    '/home/patrick/Work/Hualien0403/Dataset/20240403/data_final/TW.SHUL.00.HH2.D.2024,094,00:27:51.SAC'
)
st_hand.trim(starttime, endtime)
paz_HH = {
    'poles': [(-0.036 + 0.038j), (-0.036 - 0.038j), (-222 + 222j), (-222 - 222j)],
    'zeros': [0j, 0j],
    'sensitivity': 6.670780e08,
    'gain': 9.853340e04,
}
st_hand.simulate(paz_remove=paz_HH, pre_filt=prefilt)
st_sac_cut = read('/home/patrick/Work/test_SHUL.SAC')
st_to_wa = read('/home/patrick/Work/test_pz_SHUL.SAC')
st_to_vel = read('/home/patrick/Work/test_acc_SHUL.SAC')

# %%
from obspy import UTCDateTime, read
from obspy.io.sac.sacpz import attach_paz

## accelaration
pz_file = '/data/share/for_patrick/PZ_test/SAC_PZs_TW_SM09_HLZ_00_2021.001.00.00.00.0000_2599.365.23.59.59.99999'
sac_data = '/home/patrick/Work/TW.SM09.00.HLZ.D.2024.094.SAC'

prefilt = (0.1, 0.5, 30, 35)

# starttime = UTCDateTime('2024-04-03T03:56:19')  # 14179
# endtime = UTCDateTime('2024-04-03T03:58:19')  # 14299
# using attach_paz
st = read(sac_data)
# %%
# st.trim(starttime, endtime)
attach_paz(tr=st[0], paz_file=pz_file, tovel=True)
st[0].stats.paz['zeros'] = [i for i in st[0].stats.paz['zeros'] if i != 0j]
wa_simulate_1 = {
    'poles': [(-6.28318 + 4.71239j), (-6.28318 - 4.71239j)],
    'zeros': [0j, 0j],
    'sensitivity': 1.0,
    'gain': 2800.0,
}

wa_simulate_2 = {
    'poles': [(-5.49779 + 5.60886j), (-5.49779 - 5.60886j)],
    'zeros': [0j, 0j],
    'sensitivity': 1.0,
    'gain': 2080.0,
}
st.simulate(paz_remove='self', paz_simulate=wa_simulate_1, pre_filt=prefilt)
st_to_wa = read('/home/patrick/Work/SM09_whole_WA.SAC')
st_to_acc = read('/home/patrick/Work/SM09_whole.SAC')
# %%
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 120, 12001)
y_hand = st_hand[0].data - st_to_acc[0].data
y = st[0].data - st_to_acc[0].data
fig, ax = plt.subplots(2, 1)
ax[0].plot(x, y)
ax[0].set_title('hand - sac')
ax[1].plot(x, y_hand)
ax[1].set_title('attach - sac')

# %%
import time
from pathlib import Path

from obspy import Stream, read


def merge():
    sac_parent_dir = Path('/raid1/share/for_patrick/AutoQuake_testset/SAC/20240402')
    sac_list = list(sac_parent_dir.glob('*WUSB*Z*'))
    stream = Stream()
    for sac_file in sac_list:
        st = read(sac_file)
        stream += st
    stream = stream.merge(fill_value='latest')
    return stream


start_time = time.time()
result = merge()
end_time = time.time()

elapsed_time = end_time - start_time
print(f'My function took {elapsed_time:.6f} seconds to run.')
# %%
from obspy import read

st = read(
    '/raid1/share/for_patrick/AutoQuake_testset/SAC/20240403/TW.EOS3.20.EH1.D.2024,094,00:00:00.SAC'
)
# %%
from pathlib import Path

import numpy as np

result_path = Path('/home/patrick/Work/AutoQuake/H3DD/testset_1029.dat_ch.dout')
test = np.nan
# %%
from pathlib import Path

import pandas as pd

test = Path(
    '/home/patrick/Work/AutoQuake/testset_DAS/picks_phasenet_das/20240402/picks.csv'
)
df = pd.read_csv(test)

# %%
from pathlib import Path

from autoquake.associator import GaMMA
from autoquake.picker import PhaseNet
from autoquake.relocator import H3DD
from autoquake.scenarios import run_autoquake

if __name__ == '__main__':
    sac_parent_dir = Path('/raid1/share/for_patrick/AutoQuake_testset/SAC')
    station_sac = Path('/home/patrick/Work/EQNet/tests/hualien_0403/station_seis.csv')
    pz_dir = Path('/raid1/share/for_patrick/AutoQuake_testset/PZ_dir/')
    result_path = Path('/home/patrick/Work/AutoQuake/testset_DAS')
    h5_parent_dir = Path('/raid1/share/for_patrick/AutoQuake_testset/hdf5/')
    startdate = '20240401'
    enddate = '20240403'
    station_das = Path('/home/patrick/Work/EQNet/tests/hualien_0403/station_das.csv')
    model_3d = Path('/home/patrick/Work/AutoQuake/H3DD/tomops_H14')
    gamma_vel_model_1 = Path('/home/patrick/Work/AutoQuake/vel_model/midas_vel.csv')
    gamma_vel_model_2 = Path(
        '/home/patrick/Work/AutoQuake_pamicoding/GaMMA/Hualien_data_20240402/Hualien_1D.vel'
    )
    h5_list = Path('/home/patrick/Work/EQNet/h5_lst')

    phasenet = PhaseNet(
        data_parent_dir=h5_parent_dir,
        start_ymd=startdate,
        end_ymd=enddate,
        result_path=result_path,
        format='h5',
        model='phasenet_das',
        device='cpu',
        highpass_filter=0.0,
    )
    gamma = GaMMA(
        station=station_das,
        result_path=result_path,
        pickings=phasenet.picks,
        vel_model=gamma_vel_model_2,
        x_interval=0.1,
        y_interval=0.1,
        covariance_prior=[1000, 1000],  # try this
        ins_type='DAS',
        picking_name_extract=None,
        min_picks_per_eq=83,
        min_p_picks_per_eq=0,
        min_s_picks_per_eq=0,
        max_sigma11=2.0,  # try this
    )

    h3dd = H3DD(
        gamma_event=gamma.events,
        gamma_picks=gamma.picks,
        station=gamma.station,
        model_3d=model_3d,
        event_name=result_path.name,
    )

    run_autoquake(
        picker=phasenet, associator=gamma, relocator=h3dd, use_magnitude=False
    )
# %%
from pathlib import Path

import matplotlib.pyplot as plt

from autoquake.visualization.comp_plot import catalog_compare, pack

cwa_catalog = Path(
    '/home/patrick/Work/AutoQuake/test_format/cwa_gafocal_20240401_20240417_results.txt'
)
aq_catalog = Path(
    '/home/patrick/Work/AutoQuake/test_format/gamma_gafocal_20240401_20240417_results.txt'
)
cwa_h3dd = Path(
    '/home/patrick/Work/AutoQuake/test_format/cwa_20240401_20240417_polarity.dout'
)
aq_h3dd = Path(
    '/home/patrick/Work/AutoQuake/test_format/gamma_20240401_20240417_polarity.dout'
)


def remove_symbol(text: str) -> int:
    """
    Better removing it when creates the catalog.
    """
    return int(text.split('+')[0])


def cal_residual(df_comp, df_main):
    for df in [df_comp, df_main]:
        df[['strike', 'dip', 'rake']] = df[['strike', 'dip', 'rake']].map(remove_symbol)
    strike_residual = []
    dip_residual = []
    rake_residual = []
    for _, row in df_comp.iterrows():
        strike = abs(row['strike'] - df_main.loc[row['comp_index'], 'strike'])
        dip = abs(row['dip'] - df_main.loc[row['comp_index'], 'dip'])
        rake = abs(row['rake'] - df_main.loc[row['comp_index'], 'rake'])
        strike_residual.append(strike)
        dip_residual.append(dip)
        rake_residual.append(rake)
    return strike_residual, dip_residual, rake_residual


def find_outliers(data_list: list, outlier) -> list:
    """
    Rerturn the index of outliers in the comp_common.
    """
    return [i for i, data in enumerate(data_list) if data == outlier]


def plot_residual(strike_residual, dip_residual, rake_residual):
    data = [strike_residual, dip_residual, rake_residual]
    # Creating a box plot
    fig, ax = plt.subplots()
    bp = ax.boxplot(
        data,
        notch=True,
        patch_artist=True,
        boxprops=dict(facecolor='lightgray'),
        zorder=1,
    )
    for i in range(len(data)):
        outliers = bp['fliers'][i].get_ydata()
        for outlier in outliers:
            index = find_outliers(data[i], outlier)
            ax.annotate(
                f'event {index[0]}',
                xy=(i + 1, outlier),
                xytext=(i + 1.05, outlier),
                ha='left',
                color='red',
            )
    # Customizing the plot
    ax.set_title('Residual between AutoQuake and CWA')
    ax.set_xlabel('Focal mechanism')
    ax.set_ylabel('Residual')
    ax.set_xticks([1, 2, 3], ['Strike', 'Dip', 'Rake'])

    return ax


def post(ax):
    ax.scatter(1, 50, color='r', s=10, zorder=2)
    ax.scatter(2, 50, color='r', s=10, zorder=2)
    ax.scatter(3, 50, color='r', s=10, zorder=2)
    plt.show()


def main():
    catalog = pack(
        catalog_list=[aq_catalog, cwa_catalog], name_list=['AutoQuake', 'CWA']
    )
    aq_common, cwa_common, aq_only, cwa_only = catalog_compare(catalog=catalog, tol=5)
    strike_residual, dip_residual, rake_residual = cal_residual(aq_common, cwa_common)
    ax = plot_residual(strike_residual, dip_residual, rake_residual)
    post(ax)
    return [strike_residual, dip_residual, rake_residual]


# %%
import numpy as np

# Generating some sample data
np.random.seed(10)
data = [np.random.normal(0, std, 100) for std in range(1, 4)]

# Creating a box plot
plt.figure(figsize=(10, 6))
plt.boxplot(data, notch=True, patch_artist=True)
# Customizing the plot
plt.title('Box Plot Example with Matplotlib')
plt.xlabel('Sample Group')
plt.ylabel('Values')
plt.xticks([1, 2, 3], ['Group 1', 'Group 2', 'Group 3'])

# Display the plot
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np

# Example observed data
observed_data = [15, 18, 20, 22, 19, 24, 17, 21, 23, 20]

# Creating the box plot
plt.figure(figsize=(12, 6))
plt.boxplot(
    observed_data, notch=True, patch_artist=True
)  # , boxprops=dict(facecolor='lightgray'))

# Overlaying the individual data points directly on the same x-position (x=1)
x_position = np.ones(len(observed_data))  # Align all points at x=1
plt.scatter(
    x_position, observed_data, color='blue', alpha=0.6, label='Observed Data Points'
)

# Customizing the plot
plt.title('Box Plot with Overlayed Observed Data Points')
plt.ylabel('Value')
plt.xticks([1], ['Observed Data Distribution'])
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend()

# Display the plot
plt.show()
# %% ==== this is for selecting the station we want!
import bisect

import pandas as pd

from autoquake.visualization._plot_base import convert_channel_index

pol_picks = '/home/patrick/Work/Hualien0403/GaMMA/gamma_test/synthetic_main_eq/gamma_events_polarity_tf.csv'
station_das = '/home/patrick/Work/EQNet/tests/hualien_0403/station_das.csv'


def process_list(pol_picks, station_das):
    # Read the CSV files
    df_pol_picks = pd.read_csv(pol_picks)
    df_station = pd.read_csv(station_das)
    df_pol_picks = df_pol_picks[
        df_pol_picks['station_id'].map(lambda x: x[1].isdigit())
        & (df_pol_picks['polarity'] != 'x')
    ]
    df_pol_picks['channel'] = df_pol_picks['station_id'].map(convert_channel_index)
    df_station['channel'] = df_station['station'].map(convert_channel_index)
    return df_pol_picks, df_station


def find_closest_station(target, valid_stations):
    """
    Find the closest valid station to the target.
    """
    index = bisect.bisect_left(valid_stations, target)
    if index == 0:
        return valid_stations[0]
    if index == len(valid_stations):
        return valid_stations[-1]
    before = valid_stations[index - 1]
    after = valid_stations[index]
    return after if abs(after - target) < abs(before - target) else before


def select_stations(requested_stations, valid_stations, num_stations=20):
    """
    Selects a fixed number of stations with equal step from requested stations
    and matches them to the closest valid station.
    """
    # Sort valid stations for binary search
    valid_stations.sort()
    step = max(1, len(requested_stations) // num_stations)
    selected_stations = []

    for i in range(0, len(requested_stations), step):
        if len(selected_stations) >= num_stations:
            break
        print(requested_stations[i])
        closest_station = find_closest_station(requested_stations[i], valid_stations)
        if closest_station not in selected_stations:  # Avoid duplicates
            selected_stations.append(closest_station)

    return selected_stations


def filter_das_station(pol_picks, station_das, output_path, num_stations=20):
    df_pol_picks, df_station = process_list(pol_picks, station_das)
    requested_stations = sorted(list(df_pol_picks['channel']))
    valid_stations = sorted(list(df_station['channel']))
    selected_station = select_stations(
        requested_stations, valid_stations, num_stations=num_stations
    )
    df_test = df_station[df_station['channel'].isin(selected_station)]
    df_test.drop(columns=['channel'], inplace=True)
    df_test.to_csv(output_path, index=False)


filter_das_station(pol_picks, station_das, output_path, num_stations=20)
# %%
from pathlib import Path

from autoquake.visualization.check_plot import parallel_plot_asso

phasenet_picks = Path(
    '/home/patrick/Work/AutoQuake/test_agu/20240401_20240408/all_picks.csv'
)
# gamma_events = Path(
#     '/home/patrick/Work/AutoQuake/test_agu/seis_das_test/cov_250/gamma_event_type_3.csv'  #####
# )
gamma_picks = Path(
    '/home/patrick/Work/AutoQuake/test_agu/20240401_20240408/cov_100/gamma_picks.csv'
)
# fig_dir = Path('/home/patrick/Work/AutoQuake/test_agu/figures/gamma_event_type_3')
# station_das_20 = Path('/home/patrick/Work/Hualien0403/stations/seis_das_20.csv')
station_all = Path('/home/patrick/Work/Hualien0403/stations/station_all.csv')
sac_parent_dir = Path('/home/patrick/Work/Hualien0403/data_parent_dir')
h5_parent_dir = Path('/raid4/DAS_data/iDAS_MiDAS/hdf5')
for i in [1, 2, 3, 4]:
    gamma_events = Path(
        f'/home/patrick/Work/AutoQuake/test_agu/20240401_20240408/cov_100/gamma_events_type_{i}.csv'
    )  # noqa: E501
    fig_dir = Path(
        f'/home/patrick/Work/AutoQuake/test_agu/20240401_20240408/cov_100/figure/gamma_events_type_{i}'
    )
    parallel_plot_asso(
        phasenet_picks=phasenet_picks,
        gamma_events=gamma_events,
        gamma_picks=gamma_picks,
        station=station_all,
        fig_dir=fig_dir,
        sac_parent_dir=sac_parent_dir,
        h5_parent_dir=h5_parent_dir,
        processes=40,
    )

# plot_asso(
#     df_phasenet_picks=df_phasenet_picks,
#     gamma_picks=gamma_picks,
#     gamma_events=gamma_events,
#     station=station_all,
#     event_i=63,  #####
#     fig_dir=fig_dir,
#     sac_parent_dir=sac_parent_dir,
#     h5_parent_dir=h5_parent_dir,
# )
