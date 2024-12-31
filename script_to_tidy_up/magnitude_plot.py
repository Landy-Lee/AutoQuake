# %%
import logging
import os
import warnings

logging.basicConfig(
    filename='mag.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
)
from datetime import datetime, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator


# Read the earthquake data
def plot(catalog_path, catalog_name):
    df = pd.read_csv(catalog_path)

    # Convert date and time columns to a single datetime column
    df['datetime'] = pd.to_datetime(df['time'])

    # Plot magnitude distribution
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(df['magnitude'], bins=16, range=(0, 8), edgecolor='k')
    # Annotate the counts above each bar
    for i in range(len(n)):
        if n[i] > 0:  # Only plot text if the bin's number is greater than 0
            plt.text(
                bins[i] + (bins[i + 1] - bins[i]) / 2,
                n[i],
                str(int(n[i])),
                ha='center',
                va='bottom',
            )
    plt.legend([f'Total events: {int(n.sum())}'], loc='upper right')
    plt.title('Magnitude Distribution', fontsize=15)
    plt.xlabel('Magnitude', fontsize=15)
    plt.ylabel('Frequency', fontsize=15)
    # Add minor ticks with interval equal to 0.5
    ax = plt.gca()
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.tick_params(axis='x', which='minor', length=5)
    ax.set_xlim(left=0)
    # plt.grid(True)
    plt.savefig(
        os.path.join(parent_dir, f'{catalog_name}_mag_histogram.png'),
        dpi=300,
        bbox_inches='tight',
    )

    # Show correlation with scatter plot
    plt.figure(figsize=(15, 10))
    plt.scatter(df['datetime'], df['magnitude'], alpha=0.7)
    plt.title('Magnitude vs Date', fontsize=25, pad=10)
    plt.xlabel('Date', fontsize=25, labelpad=10)
    plt.ylabel('Magnitude', fontsize=25, labelpad=10)
    plt.grid(True)
    # Customize x-axis ticks
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.tick_params(axis='x', which='major', labelsize=12, labelrotation=45)
    ax.tick_params(
        axis='x', which='minor', labelsize=12, length=3
    )  # Adjust minor ticks length
    ax.tick_params(axis='y', which='major', labelsize=20)
    # Adjust offset of x-tick labels
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('right')

    # Set x-limits
    ax.set_xlim([start_date, end_date])
    ax.set_ylim(0, 8)
    plt.savefig(
        os.path.join(parent_dir, f'{catalog_name}_mag_daily.png'),
        dpi=300,
        bbox_inches='tight',
    )


def plot_both(catalog_list, name_list):
    plt.figure(figsize=(15, 6))
    for catalog, name in zip(catalog_list, name_list):
        df = pd.read_csv(catalog)
        df['datatime'] = pd.to_datetime(df['time'])

        # Bin magnitudes and count frequencies for both datasets
        bins = np.arange(0, 9, 1)
        df['mag_bin'] = pd.cut(df['magnitude'], bins=bins, right=False)

        counts1 = df['mag_bin'].value_counts().sort_index()

        # Create a DataFrame for plotting
        counts_df = pd.DataFrame({name: counts1}).fillna(0)

        ax = counts_df.plot(kind='bar', width=0.8)
        # Adding count labels on top of each bar
        for p in ax.patches:
            ax.annotate(
                str(int(p.get_height())),
                (p.get_x(), p.get_height() * 1.005),
                ha='left',
                va='bottom',
            )
        # Customize x-tick labels
        ax.set_xticklabels(
            [f'{bin.left:.1f}-{bin.right:.1f}' for bin in counts_df.index],
            rotation=45,
            ha='right',
            rotation_mode='anchor',
        )

    plt.title('Magnitude Distribution Comparison', fontsize=15)
    plt.xlabel('Magnitude Bins', fontsize=15)
    plt.ylabel('Frequency', fontsize=15)
    plt.legend(fontsize=15)
    plt.grid(axis='y')
    plt.savefig(
        os.path.join(parent_dir, f'{name_list[0]}_{name_list[1]}_mag_bar_chart.png'),
        dpi=300,
        bbox_inches='tight',
    )

    # Scatter plot to show magnitude vs. date with different colors for each dataset
    plt.figure(figsize=(15, 10))
    color_map = {'GDMS': 'green', 'only_seis': 'blue', 'seis_das': 'yellow'}
    for catalog, name in zip(catalog_list, name_list):
        df = pd.read_csv(catalog)
        df['datatime'] = pd.to_datetime(df['time'])
        plt.scatter(
            df['datetime'], df['magnitude'], alpha=0.7, label=name, color='darkorange'
        )

    plt.title('Magnitude vs Date', fontsize=25, pad=10)
    plt.xlabel('Date', fontsize=25, labelpad=10)
    plt.ylabel('Magnitude', fontsize=25, labelpad=10)
    plt.legend()
    plt.grid(True)

    # Customize x-axis ticks
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.tick_params(axis='x', which='major', labelsize=12, labelrotation=45)
    ax.tick_params(axis='x', which='minor', labelsize=12, length=3)
    ax.tick_params(axis='y', which='major', labelsize=20)

    # Adjust offset of x-tick labels
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('right')

    # Set x-limits
    ax.set_xlim([start_date, end_date])
    plt.savefig(
        os.path.join(parent_dir, f'{cat}_mag_daily.png'), dpi=300, bbox_inches='tight'
    )


def date_range(start_date, end_date):
    delta = end_date - start_date
    date_list = []
    for i in range(delta.days + 1):
        date = start_date + timedelta(days=i)
        date_list.append(date.strftime('%Y%m%d'))
    return date_list


# extracting N4 magnitude
def mag_to_catalog(mag_parent_dir, output_file):
    date_list = date_range(start_date, end_date)
    with open(output_file, 'w') as w:
        for date in date_list:
            mag_path = os.path.join(mag_parent_dir, date, 'mag_lst')
            with open(mag_path) as r:
                lines = r.readlines()
                for line in lines:
                    part = line.strip().split()
                    if part[0][:4] == analyze_year:
                        mag_box = []
                        date = part[0]
                        hms = part[1]
                        lon = part[3]
                        lat = part[4]
                        depth = part[5]
                    elif part[0] == 'average:':
                        with warnings.catch_warnings(record=True) as w_list:
                            warnings.simplefilter('always')
                            mag_average = np.mean(mag_box)
                            if w_list:
                                for warning in w_list:
                                    logging.warning(
                                        f'{warning} triggered in file {mag_path}: {line.strip()}'
                                    )

                        w.write(f'{date} {hms} {lat} {lon} {depth} {mag_average}\n')
                    else:
                        sta_mag = float(part[2])
                        # inspect the type
                        if sta_mag != '-Inf':
                            # inspect the magnitude
                            if float(sta_mag) > 0:
                                mag_box.append(float(sta_mag))


# Function to log warnings
def log_warning(message, category, filename, lineno, file=None, line=None):
    logging.warning(f'{filename}:{lineno}: {category.__name__}: {message}')


# Redirect warnings to the logging system
warnings.showwarning = log_warning

if __name__ == '__main__':
    analyze_year = '2024'
    start_date = datetime(2024, 4, 1)
    end_date = datetime(2024, 4, 8)
    parent_dir = (
        '/home/patrick/Work/AutoQuake/test_agu/20240401_20240408/cov_100/figure/mag'
    )
    gdms_catalog = (
        '/home/patrick/Work/AutoQuake/test_agu/CWA/gdms_detailed_0401_0408.csv'
    )
    only_seis_catalog = (
        '/home/patrick/Work/AutoQuake/test_format/gamma_detailed_20240401_20240408.csv'
    )
    seis_das_catalog = '/home/patrick/Work/AutoQuake/test_agu/20240401_20240408/cov_100/gamma_final_detailed.csv'
    os.makedirs(parent_dir, exist_ok=True)
    plot(gdms_catalog, 'GDMS')


# %%
