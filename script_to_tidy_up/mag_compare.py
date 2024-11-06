#%%
import os
import logging
import argparse
logging.basicConfig(filename='mag_check.log', level=logging.INFO, filemode='w')
import pandas as pd
import numpy as np
from obspy import UTCDateTime
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def plot(limit_100 = False):
    df_cwb = pd.read_csv(cwb_mag, sep='\s+', header=None)
    df_cwb.columns = ['date', 'time', 'latitude', 'longitude', 'depth', 'magnitude', 'other_1','other_2','other_3','other_4']
    df_cwb['index'] = df_cwb.index  # Add an index column

    df_n4 = pd.read_csv(n4_mag, sep='\s+', header=None)
    df_n4.columns = ['date', 'time', 'latitude', 'longitude', 'depth', 'magnitude']

    df_gamma = pd.read_csv(gamma_mag)

    for df in [df_cwb, df_n4]:
        df['date'] = df['date'].astype(str).apply(lambda x: f"{x[:4]}-{x[4:6]}-{x[6:]}")
        df['time'] = df['time'].astype(str).str.zfill(8).apply(lambda x: f"{x[:2]}:{x[2:4]}:{x[4:6]}.{x[6:]}")
        df['datetime'] = df['date'] + 'T' + df['time']
    
    # new dataframe for n4 and gamma
    n4_comp = pd.DataFrame(columns=['index', 'datetime','magnitude'])
    gamma_comp = pd.DataFrame(columns=['index', 'datetime','magnitude'])

    # Process the CWB and N4 data
    for _, cwb_row in df_cwb[df_cwb['date'] == '2024-04-03'].iterrows():
        cwb_time = cwb_row['datetime']
        cwb_index = cwb_row['index']
        check = False
        for _, n4_row in df_n4[df_n4['date'] == '2024-04-03'].iterrows():
            n4_time = n4_row['datetime']
            if abs(UTCDateTime(cwb_time) - UTCDateTime(n4_time)) < 5:
                # Append the row to n4_comp
                n4_comp = pd.concat([n4_comp, pd.DataFrame([[cwb_index, n4_time, n4_row['magnitude']]], columns=n4_comp.columns)], ignore_index=True)
                check = True
                break
        if not check:
            logging.info(f"{cwb_time} not found in n4_time")

    # Process the CWB and gamma data
    for _, cwb_row in df_cwb[df_cwb['date'] == '2024-04-03'].iterrows():
        cwb_time = cwb_row['datetime']
        cwb_index = cwb_row['index']
        checkk = False
        for _, gamma_row in df_gamma.iterrows():
            gamma_time = gamma_row['time']
            if abs(UTCDateTime(cwb_time) - UTCDateTime(gamma_time)) < 5:
                # Append the row to gamma_comp
                gamma_comp = pd.concat([gamma_comp, pd.DataFrame([[cwb_index, gamma_time, gamma_row['magnitude']]], columns=gamma_comp.columns)], ignore_index=True)
                checkk = True
                break
        if not checkk:
            logging.info(f"{cwb_time} not found in gamma_time")

    # Find common indices
    common_indices = set(n4_comp['index']).intersection(set(gamma_comp['index'])).intersection(set(df_cwb['index']))

    # Filter DataFrames to only include common indices
    df_cwb_common = df_cwb[df_cwb['index'].isin(common_indices)]
    n4_comp_common = n4_comp[n4_comp['index'].isin(common_indices)]
    gamma_comp_common = gamma_comp[gamma_comp['index'].isin(common_indices)]

    # Optionally limit to the first 100 common indices
    if limit_100:
        df_cwb_common = df_cwb_common.head(50)
        n4_comp_common = n4_comp_common.head(50)
        gamma_comp_common = gamma_comp_common.head(50)

    # Calculate differences
    n4_diff = df_cwb_common['magnitude'].values - n4_comp_common['magnitude'].values
    gamma_diff = df_cwb_common['magnitude'].values - gamma_comp_common['magnitude'].values
    ng_diff = n4_comp_common['magnitude'].values - gamma_comp_common['magnitude'].values

    # Calculate mean differences
    mean_n4_diff = np.mean(n4_diff)
    mean_gamma_diff = np.mean(gamma_diff)
    mean_ng_diff = np.mean(ng_diff)

    # Plot the scatter plots and the differences
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 1, 1)  # Create a subplot for scatter plot
    plt.scatter(df_cwb_common['index'], df_cwb_common['magnitude'], alpha=0.7, label='CWB', color='green', s=10)
    plt.scatter(n4_comp_common['index'], n4_comp_common['magnitude'], alpha=0.7, label='N4', color='red', s=10)
    plt.scatter(gamma_comp_common['index'], gamma_comp_common['magnitude'], alpha=0.7, label='Gamma', color='blue', s=10)

    plt.title('Magnitude Scatter Plot on 2024-04-03', fontsize=15)
    plt.xlabel('Index', fontsize=12)
    plt.ylabel('Magnitude', fontsize=12)
    plt.legend()
    plt.grid(True)

    ax = plt.gca()
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))  # Set minor y ticks
    ax.tick_params(axis='y', which='minor', length=4)

    plt.subplot(2, 1, 2)  # Create a subplot for difference plot
    plt.plot(df_cwb_common['index'], n4_diff, label='CWB vs N4', color='red', alpha=0.7)
    plt.plot(df_cwb_common['index'], gamma_diff, label='CWB vs Gamma', color='blue', alpha=0.7)
    plt.plot(df_cwb_common['index'], ng_diff, label='N4 vs Gamma', color='green', alpha=0.7)

    plt.title('Magnitude Differences on 2024-04-03', fontsize=15)
    plt.xlabel('Index', fontsize=12)
    plt.ylabel('Magnitude Difference', fontsize=12)
    plt.legend()
    plt.grid(True)
    
    ax = plt.gca()
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))  # Set minor y ticks
    ax.tick_params(axis='y', which='minor', length=4)
    # Display mean differences
    plt.text(0.5, 0.1, f'Mean CWB vs N4: {mean_n4_diff:.2f}\nMean CWB vs Gamma: {mean_gamma_diff:.2f}\nMean N4 vs Gamma: {mean_ng_diff:.2f}', 
             ha='center', va='center', transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(parent_dir, 'mag_comparison_20240403.png'), dpi=300, bbox_inches='tight')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot magnitude data.')
    parser.add_argument('--limit_100', action='store_true', help='Limit plot to the first 100 indices')
    args = parser.parse_args()

    gamma_mag = '/home/patrick/Work/AutoQuake/GaMMA/results/Hualien_0403/remove_first/gamma_events.csv'
    n4_mag = '/home/patrick/Work/playground/gamma_mag_catalog_20240401_20240417.txt'
    cwb_mag = '/home/patrick/Work/playground/cwb_mag_catalog_20240401_20240417.txt'
    parent_dir = '/home/patrick/Work/playground/fig'
    plot(limit_100=args.limit_100)