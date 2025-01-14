# import os
# import glob
# import math
import logging

# from obspy.imaging.beachball import beach
import multiprocessing as mp
import string
from functools import partial

# import calendar
# from datetime import datetime, timedelta
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from pyproj import Geod
from typing import Any
# from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
# from collections import defaultdict
# from pyrocko import moment_tensor as pmt
# from pyrocko.plot import beachball, mpl_color
from ._plot_base import (
    _find_phasenet_das_pick,
    _find_phasenet_seis_pick,
    _preprocess_focal_files,
    _preprocess_phasenet_csv,
    add_on_utc_time,
    add_vel_view,
    add_vel_profile,
    catalog_filter,
    check_format,
    find_gafocal_polarity,
    find_gamma_h3dd,
    find_gamma_index,
    find_gamma_pick,
    find_sac_data,
    get_beach,
    get_mapview,
    get_only_beach,
    get_single_beach,
    get_tracer_beach,
    plot_station,
    plot_waveform_check,
    polarity_color_select,
    preprocess_gamma_csv,
    station_mask,
    extract_topo_profile_with_geod,
    add_topo_profile
)


def _find_asso_map(df_seis_station: pd.DataFrame, df_das_station: pd.DataFrame):
    """## return a code for map setting."""
    if df_seis_station.empty:
        # typically, only DAS
        pass
    elif df_das_station.empty:
        # typically, only seismic
        pass
    else:
        # both seismic and DAS
        pass


# TODO: MODIFYING THE EVENT_DICT, NOT NECESSARY
def plot_asso_map(
    station: Path,
    event_dict: dict | None = None,
    df_event_picks: pd.DataFrame | None = None,
    main_ax: GeoAxes | None = None,  # typically seis
    sub_ax=None,  # typically das
    main_region: list | None = None,
    sub_region: list | None = None,
    use_gamma=True,
    use_h3dd=False,
    station_mask=station_mask,
):
    """
    ## This is the map for plotting association.
    """
    map_proj = ccrs.PlateCarree()
    tick_proj = ccrs.PlateCarree()
    # if main_ax is None and sub_ax is None:
    #     fig = plt.figure(figsize=(8, 12))
    # gs = GridSpec(2, 1, height_ratios=[3, 1])
    df_station = pd.read_csv(station)
    df_seis_station = df_station[df_station['station'].map(station_mask)]
    df_das_station = df_station[~df_station['station'].map(station_mask)]
    if not df_seis_station.empty:
        # get seismic station
        if main_ax is None:
            fig = plt.figure(figsize=(8, 12))
            gs = GridSpec(2, 1, height_ratios=[3, 1])
            main_ax = fig.add_subplot(gs[0], projection=map_proj)
        if main_region is None:
            main_region = [
                df_seis_station['longitude'].min() - 0.5,
                df_seis_station['longitude'].max() + 0.5,
                df_seis_station['latitude'].min() - 0.5,
                df_seis_station['latitude'].max() + 0.5,
            ]
        get_mapview(
            main_region,
            main_ax,
            title=f"GDMS: {event_dict['h3dd']['event_time']}\nLat: {event_dict['h3dd']['event_lat']} Lon: {event_dict['h3dd']['event_lon']} depth: {event_dict['h3dd']['event_depth']}",
        )
        main_ax.scatter(
            x=df_seis_station['longitude'],
            y=df_seis_station['latitude'],
            marker='^',
            color='silver',
            edgecolors='k',
            s=50,
            zorder=2,
        )
        if use_gamma and event_dict is not None:
            main_ax.scatter(
                x=event_dict['gamma']['event_lon'],
                y=event_dict['gamma']['event_lat'],
                marker='*',
                color='y',
                s=100,
                zorder=4,
                label='GaMMA',
            )
        if use_h3dd and event_dict is not None:
            main_ax.scatter(
                x=event_dict['h3dd']['event_lon'],
                y=event_dict['h3dd']['event_lat'],
                marker='*',
                color='r',
                s=100,
                zorder=4,
                label='H3DD',
            )
        main_ax.legend()
        if df_event_picks is not None:
            for sta in df_event_picks[df_event_picks['station_id'].map(station_mask)][
                'station_id'
            ].unique():
                main_ax.scatter(
                    x=df_seis_station[df_seis_station['station'] == sta]['longitude'],
                    y=df_seis_station[df_seis_station['station'] == sta]['latitude'],
                    marker='^',
                    color='darkorange',
                    edgecolors='k',
                    s=50,
                    zorder=3,
                )
        seis_gl = main_ax.gridlines(draw_labels=True)
        seis_gl.top_labels = False  # Turn off top labels
        seis_gl.right_labels = False
        main_ax.autoscale(tight=True)
        main_ax.set_aspect('auto')
    if not df_das_station.empty:
        if sub_ax is None:
            sub_ax = fig.add_subplot(gs[1], projection=map_proj)
        if sub_region is None:
            sub_region = [
                df_das_station['longitude'].min() - 0.01,
                df_das_station['longitude'].max() + 0.01,
                df_das_station['latitude'].min() - 0.01,
                df_das_station['latitude'].max() + 0.01,
            ]
        get_mapview(sub_region, sub_ax, title='')
        sub_ax.scatter(
            x=df_das_station['longitude'],
            y=df_das_station['latitude'],
            marker='.',
            color='silver',
            s=5,
            zorder=2,
        )
        # plot association
        if use_gamma and event_dict is not None:
            sub_ax.scatter(
                x=event_dict['gamma']['event_lon'],
                y=event_dict['gamma']['event_lat'],
                marker='*',
                color='y',
                s=100,
                zorder=4,
                label='GaMMA',
            )
        # plot relocation
        if use_h3dd and event_dict is not None:
            sub_ax.scatter(
                x=event_dict['h3dd']['event_lon'],
                y=event_dict['h3dd']['event_lat'],
                marker='*',
                color='r',
                s=100,
                zorder=4,
                label='H3DD',
            )
        sub_ax.legend()
        # plot association
        if df_event_picks is not None:
            for sta in df_event_picks[~df_event_picks['station_id'].map(station_mask)][
                'station_id'
            ].unique():
                sub_ax.scatter(
                    x=df_das_station[df_das_station['station'] == sta]['longitude'],
                    y=df_das_station[df_das_station['station'] == sta]['latitude'],
                    marker='.',
                    color='darkorange',
                    s=5,
                    zorder=3,
                )
        das_gl = sub_ax.gridlines(draw_labels=True)
        das_gl.top_labels = False  # Turn off top labels
        das_gl.right_labels = False
        # das_ax.autoscale(tight=True)
        # das_ax.set_aspect('auto')


def return_none_if_empty(df):
    if df.empty:
        return None
    return df


# This function currently modified by Hsi-An
def plot_asso(
    df_phasenet_picks: pd.DataFrame,
    gamma_picks: Path,
    gamma_events: Path,
    station: Path,
    event_i: int,
    fig_dir: Path,
    h3dd_hout=None,
    amplify_index=5,
    sac_parent_dir=None,
    h5_parent_dir=None,
    station_mask=station_mask,
    seis_region=None,
    das_region=None,
    pretime=-30,
    posttime=60,
):
    """
    ## Plotting the gamma and h3dd info.

    Args:
        - df_phasenet_picks (pd.DataFrame): phasenet picks.
        - gamma_picks (Path): Path to the gamma picks.
        - gamma_events (Path): Path to the gamma events.
        - station (Path): Path to the station info.
        - event_i (int): Index of the event to plot.
        - fig_dir (Path): Path to save the figure.

        - h3dd_hout (Path, optional): Path to the h3dd hout. Defaults to None.
        - amplify_index (int, optional): Amplify index for sac data. Defaults to 5.
        - sac_parent_dir (Path, optional): Path to the sac data. Defaults to None.
        - sac_dir_name (Path, optional): Name of the sac data directory. Defaults to None.
        - h5_parent_dir (Path, optional): Path to the h5 data (DAS). Defaults to None.
        - station_mask (function, optional): Function to mask the station. Defaults to station_mask.
        - seis_region (list, optional): Region for seismic plot. Defaults to None.
            - 4-element list: [min_lon, max_lon, min_lat, max_lat]
        - das_region (list, optional): Region for DAS plot. Defaults to None.
            - 4-element list: [min_lon, max_lon, min_lat, max_lat]
    """
    df_event, event_dict, df_event_picks = preprocess_gamma_csv(
        gamma_catalog=gamma_events,
        gamma_picks=gamma_picks,
        event_i=event_i,
        h3dd_hout=h3dd_hout,
    )

    if h3dd_hout is not None:
        status = 'h3dd'
        use_h3dd = True
        use_gamma = False
    else:
        status = 'gamma'
        use_h3dd = False
        use_gamma = True

    time_str = (
        event_dict[status]['event_time']
        .replace(':', '_')
        .replace('-', '_')
        .replace('.', '_')
    )
    save_name = f'event_{event_i}_{time_str}.png'

    # Check if this event has already been processed
    if (fig_dir / save_name).exists():
        logging.info(f'Skipping event {event_i}: file already exists.')
        return  # Skip processing for this event
    # figure setting
    fig = plt.figure()
    map_proj = ccrs.PlateCarree()
    # tick_proj = ccrs.PlateCarree()

    # retrieving data
    if sac_parent_dir is not None:
        sac_dict = find_sac_data(
            event_time=event_dict['gamma']['event_time'],
            date=event_dict['gamma']['date'],
            event_point=event_dict[status]['event_point'],
            station=station,
            sac_parent_dir=sac_parent_dir,
            amplify_index=amplify_index,
        )
        seis_map_ax = fig.add_axes([0.3, 0.5, 0.4, 0.8], projection=map_proj)
        seis_map_ax.set_title(
            f"Event_{event_i}: {event_dict[status]['event_time']}\nlat: {event_dict[status]['event_lat']}, lon: {event_dict[status]['event_lon']}, depth: {event_dict[status]['event_depth']} km"
        )
        seis_waveform_ax = fig.add_axes([0.82, 0.5, 0.8, 0.9])
    else:
        seis_map_ax = None
        seis_waveform_ax = None
        sac_dict = {}

    if h5_parent_dir is not None:
        das_map_ax = fig.add_axes([0.3, -0.15, 0.4, 0.55], projection=map_proj)
        das_waveform_ax = fig.add_axes([0.82, -0.15, 0.8, 0.55])  # , sharey=das_map_ax)
        if seis_map_ax is None:
            das_waveform_ax.set_title(
                f"Event_{event_i}: {event_dict[status]['event_time']}, lat: {event_dict[status]['event_lat']}, lon: {event_dict[status]['event_lon']}, depth: {event_dict[status]['event_depth']} km"
            )
    else:
        das_map_ax = None
        das_waveform_ax = None

    starttime = add_on_utc_time(df_event['time'].iloc[0], pretime)
    endtime = add_on_utc_time(df_event['time'].iloc[0], posttime)

    df_das_phasenet_picks = _find_phasenet_das_pick(
        starttime=starttime,
        endtime=endtime,
        df_picks=df_phasenet_picks,
    )
    df_seis_phasenet_picks = _find_phasenet_seis_pick(
        starttime=starttime,
        endtime=endtime,
        df_picks=df_phasenet_picks,
        sac_dict=sac_dict,
        picking_check=False,
        asso_check=True,
    )
    df_seis_gamma_picks, df_das_gamma_picks = find_gamma_pick(
        df_gamma_picks=df_event_picks,
        sac_dict=sac_dict,
        event_total_seconds=event_dict['gamma']['event_total_seconds'],
        station_mask=station_mask,
    )
    plot_waveform_check(
        sac_dict=sac_dict,
        starttime=starttime,
        end_time=endtime,
        df_seis_phasenet_picks=return_none_if_empty(df_seis_phasenet_picks),
        df_seis_gamma_picks=return_none_if_empty(df_seis_gamma_picks),
        df_das_phasenet_picks=return_none_if_empty(df_das_phasenet_picks),
        df_das_gamma_picks=return_none_if_empty(df_das_gamma_picks),
        h5_parent_dir=h5_parent_dir,
        das_ax=das_waveform_ax,
        seis_ax=seis_waveform_ax,
    )

    plot_asso_map(
        station=station,
        event_dict=event_dict,
        df_event_picks=df_event_picks,
        main_ax=seis_map_ax,
        sub_ax=das_map_ax,
        use_gamma=use_gamma,
        use_h3dd=use_h3dd,
        station_mask=station_mask,
        main_region=seis_region,
        sub_region=das_region,
    )

    plt.tight_layout()
    plt.savefig(fig_dir / save_name, bbox_inches='tight', dpi=300)
    plt.close()


def parallel_plot_asso(
    phasenet_picks: Path,
    gamma_events: Path,
    gamma_picks: Path,
    station: Path,
    fig_dir: Path,
    h3dd_hout: Path,
    sac_parent_dir: Path,
    h5_parent_dir: Path | None = None,
    processes=10,
):
    fig_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=fig_dir / 'aso.log',
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    df_events = pd.read_csv(gamma_events)
    event_list = list(df_events['event_index'])
    df_phasenet_picks = _preprocess_phasenet_csv(
        phasenet_picks, get_station=lambda x: x
    )
    partial_plot_asso = partial(
        plot_asso,
        h3dd_hout=h3dd_hout,
        sac_parent_dir=sac_parent_dir,
        h5_parent_dir=h5_parent_dir,
    )
    with mp.Pool(processes=processes) as pool:
        pool.starmap(
            partial_plot_asso,
            [
                (
                    df_phasenet_picks,
                    gamma_picks,
                    gamma_events,
                    station,
                    event_i,
                    fig_dir,
                )
                for event_i in event_list
            ],
        )


def _add_epicenter(
    x: float,
    y: float,
    ax,
    size=10,
    color='yellow',
    markeredgecolor='black',
    label='Epicenter',
    alpha=0.5,
):
    """## adding epicenter."""
    ax.plot(
        x,
        y,
        '*',
        markersize=size,
        color=color,
        markeredgecolor=markeredgecolor,
        label=label,
        alpha=alpha,
        zorder=10,
    )


def _add_sub_map(
    geo_ax: GeoAxes,
    epi_lon: float,
    epi_lat: float,
    epi_size=10,
    epi_color='yellow',
    alpha=0.5,
):
    """## Add sub map on the right top of the main_ax
    Only thing needs to care about is the `geo_ax`, location is important.
    #TODO: Using a smart way to automatically decide the location of the sub map.
    """
    geo_ax.coastlines(resolution='10m', color='black', linewidth=1, alpha=alpha)
    geo_ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=alpha)
    geo_ax.add_feature(cfeature.LAND, alpha=alpha)

    # epicenter
    _add_epicenter(x=epi_lon, y=epi_lat, ax=geo_ax, size=epi_size, color=epi_color)

    geo_ax.patch.set_alpha(alpha)
    geo_ax.set_aspect('auto')


def _add_polarity(
    df_station, df_polarity_picks, ax, x_coord, y_coord, marker='.', markersize=5
):
    """## Adding polarity information on profile"""
    df_intersection = pd.merge(df_station, df_polarity_picks, on='station', how='inner')
    df_intersection['color'] = df_intersection['polarity'].map(polarity_color_select)
    ax.scatter(
        df_intersection[x_coord],
        df_intersection[y_coord],
        c=df_intersection['color'],
        marker=marker,
        s=markersize,
    )


def get_polarity_map(
    geo_ax,
    region: list,
    df_station: pd.DataFrame,
    df_polarity_picks: pd.DataFrame,
    epi_lon: float,
    epi_lat: float,
    title='Polarity map',
    station_mask=station_mask,
):
    """## The map for polarity"""
    df_seis_station = df_station[df_station['station'].map(station_mask)]
    df_das_station = df_station[~df_station['station'].map(station_mask)]
    # Cartopy setting
    get_mapview(region=region, ax=geo_ax, title=title)

    # plotting station on map
    plot_station(df_station=df_station, geo_ax=geo_ax)

    # adding epicenter
    _add_epicenter(x=epi_lon, y=epi_lat, ax=geo_ax)

    # adding polarity information on it
    _add_polarity(
        df_station=df_seis_station,
        df_polarity_picks=df_polarity_picks,
        ax=geo_ax,
        x_coord='longitude',
        y_coord='latitude',
        marker='^',
        markersize=40,
    )

    _add_polarity(
        df_station=df_das_station,
        df_polarity_picks=df_polarity_picks,
        ax=geo_ax,
        x_coord='longitude',
        y_coord='latitude',
    )
    geo_ax.set_xlim(region[0], region[1])
    geo_ax.set_ylim(region[2], region[3])
    # geo_ax.autoscale(tight=True)
    geo_ax.set_aspect('auto')
    geo_ax.legend(markerscale=2, labelspacing=1.2, borderpad=1, fontsize=12)


def get_profile(
    df_station,
    ax,
    df_polarity_picks: pd.DataFrame,
    depth_lim: tuple | None = None,
    xlim: tuple | None = None,
    ylim: tuple | None = None,
    depth_axis='x',
    markersize=5,
    grid_alpha=0.7,
):
    """
    The df_polarity_picks is already filtered by the event index.
    """
    # ax.autoscale(tight=True)
    if depth_lim is None:
        depth_min = max(df_station['elevation'])
        depth_max = min(df_station['elevation'])
        depth_lim = (depth_min + 10, depth_max - 10)
    # axes setting
    # Scenario 1: depth_axis is in x cooridnate
    if depth_axis == 'x':
        x_coord = 'elevation'
        y_coord = 'latitude'
        ax.set_ylim(ylim)
        ax.set_xlim(depth_lim)
        ax.set_xlabel('Elevation (km)')
        ax.tick_params(axis='y', labelleft=False)
    elif depth_axis == 'y':
        x_coord = 'longitude'
        y_coord = 'elevation'
        ax.set_xlim(xlim)
        ax.set_ylim(depth_lim)
        ax.invert_yaxis()
        ax.set_ylabel('Elevation (km)')
        ax.tick_params(axis='x', labelbottom=False)

    ax.grid(True, alpha=grid_alpha)
    ax.scatter(df_station[x_coord], df_station[y_coord], c='silver', s=markersize)

    # plot station with polarity
    _add_polarity(
        df_station=df_station,
        df_polarity_picks=df_polarity_picks,
        ax=ax,
        x_coord=x_coord,
        y_coord=y_coord,
        markersize=markersize,
    )


def get_polarity_profiles(
    station: Path,
    gamma_events: Path,
    h3dd_events: Path,
    polarity_picks: Path,
    polarity_dout: Path,
    focal_catalog: Path,
    figure_dir: Path,
    event_index: int,
    region: list | None = None,
    depth_lim: tuple | None = None,
    focal_xlim=(-1.1, 1.1),
    focal_ylim=(-1.1, 1.1),
    focal_add_info=False,
    focal_only_das=False,
    savefig=True,
):
    """## Plot the polarity information on the station, typically for DAS.

    ### TODO: Optimization index
    This plot do not need the gamma as input, using h3dd row index to find the event.
    Main purpose is to plot the polarity of station, and find if this event has a
    solution of focal mechanism.

    ### Usage
    no need to think too much, just follow the hint and input your Path.
    """
    figure_dir.mkdir(parents=True, exist_ok=True)
    # data preprocessing
    df_h3dd = find_gamma_h3dd(gamma_events, h3dd_events, event_index)
    df_station = pd.read_csv(station)
    if region is None:
        region = [
            df_station['longitude'].min(),
            df_station['longitude'].max(),
            df_station['latitude'].min(),
            df_station['latitude'].max(),
        ]

    df_polarity_picks = pd.read_csv(polarity_picks)
    df_polarity_picks = df_polarity_picks[
        df_polarity_picks['event_index'] == event_index
    ]
    df_polarity_picks = df_polarity_picks.rename(columns={'station_id': 'station'})

    df_gafocal, df_hout = _preprocess_focal_files(
        gafocal_txt=focal_catalog, polarity_dout=polarity_dout
    )

    focal_dict = find_gafocal_polarity(
        df_gafocal=df_gafocal,
        df_hout=df_hout,
        polarity_dout=polarity_dout,
        event_index=0,  # TODO: this need to change
    )

    # The frame of plot
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 14),
        gridspec_kw={'width_ratios': [2, 1], 'height_ratios': [2, 1]},
    )

    geo_ax = fig.add_subplot(2, 2, 1, projection=ccrs.PlateCarree())
    geo_ax1 = fig.add_subplot(4, 4, 1, projection=ccrs.PlateCarree())

    # Main map
    get_polarity_map(
        geo_ax=geo_ax,
        df_station=df_station,
        df_polarity_picks=df_polarity_picks,
        region=region,
        title=f"{focal_dict['utc_time']}\nLat: {focal_dict['latitude']}, Lon: {focal_dict['longitude']}, Depth: {focal_dict['depth']}",
        epi_lon=df_h3dd['longitude'].iloc[0],
        epi_lat=df_h3dd['latitude'].iloc[0],
    )

    # Adding sub map top left
    _add_sub_map(
        geo_ax1,
        epi_lon=df_h3dd['longitude'].iloc[0],
        epi_lat=df_h3dd['latitude'].iloc[0],
    )  # 121.58, 23.86)
    plot_station(df_station, geo_ax1)

    xlim = geo_ax.get_xlim()
    ylim = geo_ax.get_ylim()

    get_profile(
        df_station=df_station,
        ax=axes[0, 1],
        df_polarity_picks=df_polarity_picks,
        ylim=ylim,
        depth_axis='x',
        depth_lim=depth_lim,
    )

    get_profile(
        df_station=df_station,
        ax=axes[1, 0],
        df_polarity_picks=df_polarity_picks,
        xlim=xlim,
        depth_axis='y',
        depth_lim=depth_lim,
    )

    get_beach(
        focal_dict=focal_dict,
        ax=axes[1, 1],
        xlim=focal_xlim,
        ylim=focal_ylim,
        only_das=focal_only_das,
        add_info=focal_add_info,
    )
    axes[0, 0].axis('off')

    # plt.tight_layout()
    if savefig:
        plt.savefig(
            figure_dir / 'pol_map.png', dpi=300, bbox_inches='tight'
        )  # using fig.savefig when we have several figs.


size_mapping = {
    (0, 1): 1,
    (1, 2): 1,
    (2, 3): 1,
    (3, 4): 1,
    (4, 5): 5,
    (5, 6): 7,
    (6, 7): 10,
    (7, 8): 15,
}
legend_elements = [
    Line2D(
        [0],
        [0],
        marker='o',
        color='w',
        label='nan',
        markersize=1,
        markerfacecolor='gray',
    ),
    Line2D(
        [0],
        [0],
        marker='o',
        color='w',
        label='0-1',
        markersize=4,
        markerfacecolor='gray',
    ),
    Line2D(
        [0],
        [0],
        marker='o',
        color='w',
        label='1-2',
        markersize=5,
        markerfacecolor='gray',
    ),
    Line2D(
        [0],
        [0],
        marker='o',
        color='w',
        label='2-3',
        markersize=6,
        markerfacecolor='gray',
    ),
    Line2D(
        [0],
        [0],
        marker='o',
        color='w',
        label='3-4',
        markersize=10,
        markerfacecolor='gray',
    ),
    Line2D(
        [0],
        [0],
        marker='o',
        color='w',
        label='4-5',
        markersize=15,
        markerfacecolor='gray',
    ),
    Line2D(
        [0],
        [0],
        marker='o',
        color='w',
        label='5-6',
        markersize=20,
        markerfacecolor='gray',
    ),
    Line2D(
        [0],
        [0],
        marker='o',
        color='w',
        label='6-7',
        markersize=25,
        markerfacecolor='gray',
    ),
    Line2D(
        [0],
        [0],
        marker='o',
        color='w',
        label='7-8',
        markersize=30,
        markerfacecolor='gray',
    ),
]


# Function to map values to sizes
def _get_size(value):
    for range_tuple, size in size_mapping.items():
        if range_tuple[0] <= value < range_tuple[1]:
            return size
    return 1


def plot_mapview(
    station: Path,
    catalog: pd.DataFrame,
    catalog_filter_: dict | None = None,
    title='Event distribution',
    main_eq=(),
    h3dd_main_eq=(),
    main_eq_size=10,
    plot_station_name=True,
    plot_event_name=True,
    station_mask=station_mask,
    savefig=True,
    fig_dir: Path | None = None,
    plot_profile=True,
    h3dd_mode=False,
    prefix='',
):
    """
    catalog should at least contains the lon and lat.

    """
    # plot setting
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 14),
        gridspec_kw={'width_ratios': [3, 1], 'height_ratios': [3, 1]},
    )
    geo_ax = fig.add_subplot(2, 2, 1, projection=ccrs.PlateCarree())
    # fig, geo_ax = plt.subplots(
    #     1, 1, figsize=(12, 14), subplot_kw={'projection': ccrs.PlateCarree()}
    # )

    # sub_ax = fig.add_subplot(4, 4, 1, projection=ccrs.PlateCarree())

    # base_map
    if catalog_filter_ is None:
        region = [
            catalog['longitude'].min() - 0.2,
            catalog['longitude'].max() + 0.2,
            catalog['latitude'].min() - 0.2,
            catalog['latitude'].max() + 0.2,
        ]
    else:
        region = [
            catalog_filter_['min_lon'],
            catalog_filter_['max_lon'],
            catalog_filter_['min_lat'],
            catalog_filter_['max_lat'],
        ]
    get_mapview(region=region, ax=geo_ax, title=title)
    # ax setting
    geo_ax.set_xlim(region[0], region[1])
    geo_ax.set_ylim(region[2], region[3])
    # trim the catalog
    catalog = catalog_filter(
        catalog_df=catalog, h3dd_mode=h3dd_mode, catalog_range=catalog_filter_
    )
    # get the size corresponding to magnitude
    catalog.loc[:, 'size'] = catalog['magnitude'].apply(_get_size)
    # station
    df_station = pd.read_csv(station)
    # print(
    #     'Geo Axes Patch Extent:',
    #     geo_ax.get_window_extent(renderer=fig.canvas.get_renderer()),
    # )
    plot_station(
        df_station=df_station,
        geo_ax=geo_ax,
        region=region,
        plot_station_name=plot_station_name,
        text_dist=0.005,
    )
    manual_elements = [
        Line2D(
            [0],
            [0],
            marker='.',
            color='w',
            label='DAS',
            markersize=5,
            markerfacecolor='k',
            markeredgecolor='k',
        ),
        Line2D(
            [0],
            [0],
            marker='^',
            color='w',
            label='Seismometer',
            markersize=7,
            markerfacecolor='c',
            markeredgecolor='k',
        ),
        Line2D(
            [0],
            [0],
            marker='*',
            color='w',
            label='CWA Epicenter',
            markersize=10,
            markerfacecolor='y',
            markeredgecolor='k',
        ),
        Line2D(
            [0],
            [0],
            marker='*',
            color='w',
            label='AutoQuake Epicenter',
            markersize=10,
            markerfacecolor='darkorange',
            markeredgecolor='k',
        ),
    ]
    # add all events
    event_num = len(catalog)
    if h3dd_mode:
        prefix = 'h3dd_'
    if 'event_type' in catalog.columns:
        # This part we want to visualize the distribution of event_type.
        color_map = {
            0: 'green',
            1: 'red',
            2: 'blue',
            3: 'green',
            4: 'orange',
            5: 'blue',
            6: 'red',
        }
        label_map = {
            0: 'seis 6P2S\n(only using seismometer)',
            1: 'seis 6P2S + DAS 15P',
            2: 'seis 6P2S',
            3: 'DAS 15P',
            4: 'Not reach the standard',
            5: 'GDMS Catalog',
            6: 'Associated event',
        }

        for event_type, group in catalog.groupby('event_type'):
            num = len(group)
            group = group.sort_values(by='size', ascending=True)
            geo_ax.scatter(
                group[f'{prefix}longitude'],
                group[f'{prefix}latitude'],
                s=group['size'],
                c=color_map[event_type],
                alpha=0.5,
                label=f'{label_map[event_type]}: {num}',
                rasterized=True,
                zorder=3 + abs(int(event_type)),
            )
            if plot_event_name:
                for _, row in group.iterrows():
                    geo_ax.text(
                        x=row[f'{prefix}longitude'],
                        y=row[f'{prefix}latitude'],
                        s=row['event_index'],
                    )
            if plot_profile:
                axes[0, 1].scatter(
                    group[f'{prefix}depth_km'],
                    group[f'{prefix}latitude'],
                    s=group['size'],
                    c=color_map[event_type],
                    alpha=0.5,
                    # label=label_map[event_type],
                    rasterized=True,
                    zorder=3 + abs(int(event_type) - 4),
                )
                _add_epicenter(
                    x=main_eq[2],
                    y=main_eq[1],
                    ax=axes[0, 1],
                    size=main_eq_size,
                    alpha=0.7,
                )
                _add_epicenter(
                    x=h3dd_main_eq[2],
                    y=h3dd_main_eq[1],
                    ax=axes[0, 1],
                    size=main_eq_size,
                    alpha=0.7,
                    color='darkorange',
                )
                axes[1, 0].scatter(
                    group[f'{prefix}longitude'],
                    group[f'{prefix}depth_km'],
                    s=group['size'],
                    c=color_map[event_type],
                    alpha=0.5,
                    # label=label_map[event_type],
                    rasterized=True,
                    zorder=3 + abs(int(event_type) - 4),
                )
                _add_epicenter(
                    x=main_eq[0],
                    y=main_eq[2],
                    ax=axes[1, 0],
                    size=main_eq_size,
                    alpha=0.7,
                )
                _add_epicenter(
                    x=h3dd_main_eq[0],
                    y=h3dd_main_eq[2],
                    ax=axes[1, 0],
                    size=main_eq_size,
                    alpha=0.7,
                    color='darkorange',
                )
            manual_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker='o',
                    color='w',
                    label=f'{label_map[event_type]}: {num}',
                    markersize=3,
                    markerfacecolor=color_map[event_type],
                    markeredgecolor='k',
                    alpha=0.5,
                )
            )
    else:
        geo_ax.scatter(
            catalog[f'{prefix}longitude'],
            catalog[f'{prefix}latitude'],
            s=group['size'],
            c='r',
            alpha=0.5,
            label=f'Event num: {event_num}',
            zorder=3,
        )
        if plot_profile:
            axes[0, 1].scatter(
                catalog[f'{prefix}depth_km'],
                catalog[f'{prefix}latitude'],
                s=5,
                c='r',
                alpha=0.5,
                rasterized=True,
            )
            axes[1, 0].scatter(
                catalog[f'{prefix}longitude'],
                catalog[f'{prefix}depth_km'],
                s=5,
                c='r',
                alpha=0.5,
                rasterized=True,
            )
    # add main eq
    _add_epicenter(x=main_eq[0], y=main_eq[1], ax=geo_ax, size=main_eq_size, alpha=0.7)
    _add_epicenter(
        x=h3dd_main_eq[0],
        y=h3dd_main_eq[1],
        ax=geo_ax,
        size=main_eq_size,
        alpha=0.7,
        color='darkorange',
    )

    # geo_ax.autoscale(tight=True)
    geo_ax.set_aspect('auto')
    geo_ax.legend(
        handles=manual_elements,  # + legend_elements,
        loc='upper left',
        markerscale=2,
        labelspacing=1.5,
        borderpad=1,
        fontsize=10,
        framealpha=0.6,
    )
    # geo_ax.add_artist(legend)
    xlim = geo_ax.get_xlim()
    ylim = geo_ax.get_ylim()
    axes[0, 1].autoscale(tight=True)
    axes[0, 1].set_ylim(ylim)
    axes[0, 1].set_xlim([0, catalog['depth_km'].max() + 1])
    # axes[0, 1].set_xlim([0, 62])  # gdms
    axes[0, 1].set_xlabel('Depth (km)')
    axes[0, 1].set_ylabel('Latitude')

    axes[1, 0].autoscale(tight=True)
    axes[1, 0].set_xlim(xlim)
    axes[1, 0].set_ylim([0, catalog['depth_km'].max() + 1])
    # axes[1, 0].set_ylim([0, 62])  # gdms
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_ylabel('Depth (km)')
    axes[1, 0].set_xlabel('Longitude')
    axes[1, 1].axis('off')
    axes[0, 0].axis('off')
    plt.tight_layout()
    if savefig and fig_dir is not None:
        plt.savefig(fig_dir / f'{title}.png', dpi=300, bbox_inches='tight')


def plot_mapview_temp(
    station: Path,
    catalog: pd.DataFrame,
    catalog_filter_: dict | None = None,
    title='Event distribution',
    eq_list=[],
    main_eq_size=10,
    plot_station_name=True,
    plot_event_name=False,
    # station_mask=station_mask,
    savefig=True,
    fig_dir: Path | None = None,
    plot_profile=True,
    h3dd_mode=False,
    prefix='',
    savename='',
    temporal=False,
    plot_profile_line=False,
    start_point=(121.08, 23.65),
    plot_vel=False,
    vel_df: pd.DataFrame | None = None # we should write a function to check the type
):
    """
    catalog should at least contains the lon and lat.

    """
    # plot setting
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 14),
        gridspec_kw={'width_ratios': [3, 1], 'height_ratios': [3, 1]},
    )
    geo_ax = fig.add_subplot(2, 2, 1, projection=ccrs.PlateCarree())
    # fig, geo_ax = plt.subplots(
    #     1, 1, figsize=(12, 14), subplot_kw={'projection': ccrs.PlateCarree()}
    # )

    # sub_ax = fig.add_subplot(4, 4, 1, projection=ccrs.PlateCarree())

    # base_map
    if catalog_filter_ is None:
        region = [
            catalog['longitude'].min() - 0.2,
            catalog['longitude'].max() + 0.2,
            catalog['latitude'].min() - 0.2,
            catalog['latitude'].max() + 0.2,
        ]
    else:
        region = [
            catalog_filter_['min_lon'],
            catalog_filter_['max_lon'],
            catalog_filter_['min_lat'],
            catalog_filter_['max_lat'],
        ]
    get_mapview(region=region, ax=geo_ax, title=title)
    if plot_vel and vel_df is not None:
        add_vel_view(
            ax=geo_ax,
            vel_df=vel_df,
            region=region
        )
    # ax setting
    geo_ax.set_xlim(region[0], region[1])
    geo_ax.set_ylim(region[2], region[3])
    # trim the catalog
    catalog = catalog_filter(
        catalog_df=catalog, h3dd_mode=h3dd_mode, catalog_range=catalog_filter_
    )
    # get the size corresponding to magnitude
    catalog.loc[:, 'size'] = catalog['magnitude'].apply(_get_size)
    # station
    df_station = pd.read_csv(station)
    # print(
    #     'Geo Axes Patch Extent:',
    #     geo_ax.get_window_extent(renderer=fig.canvas.get_renderer()),
    # )
    plot_station(
        df_station=df_station,
        geo_ax=geo_ax,
        region=region,
        plot_station_name=plot_station_name,
        text_dist=0.005,
    )
    manual_elements = [
        Line2D(
            [0],
            [0],
            marker='^',
            color='w',
            label='Seismometer',
            markersize=7,
            markerfacecolor='c',
            markeredgecolor='k',
        ),
    ]
    # add all events
    event_num = len(catalog)
    manual_elements.append(
        Line2D(
            [0],
            [0],
            marker='o',
            color='w',
            label=f'Event num: {event_num}',
            markersize=5,
            markerfacecolor='k',
            markeredgecolor='k',
        )
    )       
    if h3dd_mode:
        prefix = 'h3dd_'
    if 'event_type' in catalog.columns:
        # This part we want to visualize the distribution of event_type.
        color_map = {
            0: 'green',
            1: 'purple',
            2: 'black',
            3: 'black',
            4: 'orange',
            5: 'blue',
            6: 'red',
        }
        label_map = {
            0: 'After Main shock (2024-04-02T23:58:10)',
            1: 'Before Main shock (2024-04-02T23:58:10)',
            2: '2018 Hualien Earthquake',
            3: 'AutoQuake Catalog',
            4: 'GDMS Catalog',
            5: 'GDMS Catalog',
            6: 'Associated event',
        }

        for event_type, group in catalog.groupby('event_type'):
            num = len(group)
            group = group.sort_values(by='size', ascending=True)
            geo_ax.scatter(
                group[f'{prefix}longitude'],
                group[f'{prefix}latitude'],
                s=group['size'],
                c=color_map[event_type],
                alpha=0.5,
                label=f'{label_map[event_type]}: {num}',
                rasterized=True,
                zorder=3 + abs(int(event_type) - 4),
            )
            if plot_event_name:
                for _, row in group.iterrows():
                    geo_ax.text(
                        x=row[f'{prefix}longitude'],
                        y=row[f'{prefix}latitude'],
                        s=row['event_index'],
                    )
            if plot_profile:
                axes[0, 1].scatter(
                    group[f'{prefix}depth_km'],
                    group[f'{prefix}latitude'],
                    s=group['size'],
                    c=color_map[event_type],
                    alpha=0.5,
                    # label=label_map[event_type],
                    rasterized=True,
                    zorder=3 + abs(int(event_type) - 4),
                )
                for eq in eq_list:
                    _add_epicenter(
                        x=eq[2],
                        y=eq[1],
                        ax=axes[0, 1],
                        size=main_eq_size,
                        alpha=0.7,
                        color=eq[4],
                    )
                axes[1, 0].scatter(
                    group[f'{prefix}longitude'],
                    group[f'{prefix}depth_km'],
                    s=group['size'],
                    c=color_map[event_type],
                    alpha=0.5,
                    # label=label_map[event_type],
                    rasterized=True,
                    zorder=3 + abs(int(event_type) - 4),
                )
                for eq in eq_list:
                    _add_epicenter(
                        x=eq[0],
                        y=eq[2],
                        ax=axes[1, 0],
                        size=main_eq_size,
                        alpha=0.7,
                        color=eq[4],
                    )
            manual_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker='o',
                    color='w',
                    label=f'{label_map[event_type]}: {num}',
                    markersize=3,
                    markerfacecolor=color_map[event_type],
                    markeredgecolor='k',
                    alpha=0.5,
                )
            )

    else:
        if temporal:
            colormap = plt.cm.viridis
            norm = mcolors.Normalize(
                vmin=catalog['day'].min(),
                vmax=catalog['day'].max()
                )  # Normalize days to range [0, 1]
            for day, group in catalog.groupby('day'):
                print(f'day {day}: {len(group)}')
                color = colormap(norm(day))
                geo_ax.scatter(
                    group[f'{prefix}longitude'],
                    group[f'{prefix}latitude'],
                    s=group['size'],
                    c=color,
                    alpha=0.5,
                    label=f'Day {day} event num: {len(group)}',
                    zorder=3,
                )
                if plot_profile:
                    axes[0, 1].scatter(
                        group[f'{prefix}depth_km'],
                        group[f'{prefix}latitude'],
                        s=group['size'],
                        c=color,
                        alpha=0.5,
                        rasterized=True,
                    )
                    axes[1, 0].scatter(
                        group[f'{prefix}longitude'],
                        group[f'{prefix}depth_km'],
                        s=group['size'],
                        c=color,
                        alpha=0.5,
                        rasterized=True,
                    )
                # Add a colorbar to indicate progression
            sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
            sm.set_array([])
            # Create an inset for the colorbar within geo_ax
            cbar_ax = inset_axes(
                geo_ax,  # Parent axis (geo_ax)
                width="3%",  # Width of the colorbar relative to geo_ax
                height="50%",  # Height of the colorbar relative to geo_ax
                loc="lower right",  # Location inside the geo_ax
                bbox_to_anchor=(0.05, 0.05, 0.85, 0.9),  # Adjust position within geo_ax
                bbox_transform=geo_ax.transAxes,  # Use geo_ax coordinates
                borderpad=0,  # Padding around the inset
            )
            cbar = plt.colorbar(sm, cax=cbar_ax, orientation='vertical')
            cbar.set_label("Day Since 0401")
        else:
            geo_ax.scatter(
                catalog[f'{prefix}longitude'],
                catalog[f'{prefix}latitude'],
                s=catalog['size'],
                c='k',
                alpha=0.5,
                zorder=3,
            )         
            if plot_profile:
                axes[0, 1].scatter(
                    catalog[f'{prefix}depth_km'],
                    catalog[f'{prefix}latitude'],
                    s=catalog['size'],
                    c='k',
                    alpha=0.5,
                    rasterized=True,
                )
                axes[1, 0].scatter(
                    catalog[f'{prefix}longitude'],
                    catalog[f'{prefix}depth_km'],
                    s=catalog['size'],
                    c='k',
                    alpha=0.5,
                    rasterized=True,
                    )
        # profile
        if plot_profile_line:
            letters = string.ascii_uppercase
            for i, (dis, letter) in enumerate(zip(np.linspace(0, 100, 21), letters[0:21])):
                geod = Geod(ellps='WGS84')
                next_lon, next_lat, _ = geod.fwd(start_point[0], start_point[1], 20, dis*1000)
                text_lon, text_lat, _ = geod.fwd(start_point[0], start_point[1], 20, dis*1000)
                add_profile_line(geo_ax=geo_ax, extend_km=80, start_point=(next_lon, next_lat), azimuth=20 + 90, alpha=0.7, c='k')   
                geo_ax.text(x=text_lon -0.02, y=text_lat, s=f"{letter}-{letter}'")
            start_point = (121.485, 24.667)
            for i, (dis, letter) in enumerate(zip(np.linspace(0, 70, 15), letters[0:15])):
                geod = Geod(ellps='WGS84')
                next_lon, next_lat, _ = geod.fwd(start_point[0], start_point[1], 110, dis*1000)
                text_lon, text_lat, _ = geod.fwd(start_point[0], start_point[1], 110, dis*1000)
                add_profile_line(geo_ax=geo_ax, extend_km=120, start_point=(next_lon, next_lat), azimuth=110 + 90, alpha=0.7, c='k')   
                geo_ax.text(x=text_lon -0.02, y=text_lat, s=f"{letter}-{letter}'")                    

    # add main eq
    for eq in eq_list:
        _add_epicenter(
            x=eq[0], y=eq[1], ax=geo_ax, size=main_eq_size, alpha=0.7, color=eq[4]
        )
        _add_epicenter(
            x=eq[2], y=eq[1], ax=axes[0, 1], size=main_eq_size, alpha=0.7, color=eq[4]
        )
        _add_epicenter(
            x=eq[0], y=eq[2], ax=axes[1, 0], size=main_eq_size, alpha=0.7, color=eq[4]
        )                
        manual_elements.append(
            Line2D(
                [0],
                [0],
                marker='*',
                color='w',
                label=eq[3],
                markersize=10,
                markerfacecolor=eq[4],
                markeredgecolor='k',
            )
        )
    # geo_ax.autoscale(tight=True)
    geo_ax.set_aspect('auto')
    geo_ax.legend(
        handles=manual_elements,  # + legend_elements,
        loc='upper left',
        markerscale=2,
        labelspacing=1.5,
        borderpad=1,
        fontsize=10,
        framealpha=0.6,
    )
    # geo_ax.add_artist(legend)
    xlim = geo_ax.get_xlim()
    ylim = geo_ax.get_ylim()
    axes[0, 1].autoscale(tight=True)
    axes[0, 1].set_ylim(ylim)
    axes[0, 1].set_xlim([0, catalog['depth_km'].max() + 1])
    # axes[0, 1].set_xlim([0, 62])  # gdms
    axes[0, 1].set_xlabel('Depth (km)')
    axes[0, 1].set_ylabel('Latitude')

    axes[1, 0].autoscale(tight=True)
    axes[1, 0].set_xlim(xlim)
    axes[1, 0].set_ylim([0, catalog['depth_km'].max() + 1])
    # axes[1, 0].set_ylim([0, 62])  # gdms
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_ylabel('Depth (km)')
    axes[1, 0].set_xlabel('Longitude')
    axes[1, 1].axis('off')
    axes[0, 0].axis('off')
    plt.tight_layout()
    if savefig and fig_dir is not None:
        plt.savefig(fig_dir / f'{savename}.png', dpi=300, bbox_inches='tight')


# plot single beachball check
def plot_only_beach(
    gafocal_txt: Path,
    polarity_dout: Path,
    output_dir: Path,
    hout_file: Path | None = None,
    gamma_events: Path | None = None,
    gamma_detailed: Path | None = None,
    get_waveform=False,
    sac_parent_dir: Path | None = None,
    h5_parent_dir: Path | None = None,
    processes=10,
    event_list=[],
):
    """## plot beachball only"""
    df_gafocal, df_hout = _preprocess_focal_files(
        gafocal_txt=gafocal_txt, polarity_dout=polarity_dout, hout_file=hout_file
    )
    if len(event_list) == 0:
        event_list = range(len(df_gafocal))
    with mp.Pool(processes=processes) as pool:
        pool.starmap(
            get_single_beach,
            [
                (
                    df_gafocal,
                    polarity_dout,
                    df_hout,
                    event_index,
                    output_dir,
                    gamma_events,
                    gamma_detailed,
                    get_waveform,
                    sac_parent_dir,
                    h5_parent_dir,
                )
                for event_index in event_list
            ],
        )


# plot single beachball check
def plot_std_beach(
    gafocal_txt: Path,
    polarity_dout: Path,
    output_dir: Path,
    hout_file: Path | None = None,
    name='GAFocal',
    processes=10,
    event_list=[],
):
    """## plot beachball only"""
    df_gafocal, df_hout = _preprocess_focal_files(
        gafocal_txt=gafocal_txt, polarity_dout=polarity_dout, hout_file=hout_file
    )
    if len(event_list) == 0:
        event_list = range(len(df_gafocal))
    with mp.Pool(processes=processes) as pool:
        pool.starmap(
            get_only_beach,
            [
                (df_gafocal, polarity_dout, df_hout, event_index, output_dir, name)
                for event_index in event_list
            ],
        )


# plot beachball with tracer
def plot_tracer_beach(
    gafocal_txt: Path,
    polarity_dout: Path,
    output_dir: Path,
    tt_table: Path,
    extend_pol_picks: Path,
    station: Path,
    gamma_events: Path | None = None,
    gamma_detailed: Path | None = None,
    get_waveform=False,
    sac_parent_dir: Path | None = None,
    h5_parent_dir: Path | None = None,
    processes=10,
    input_tracer=True,
    event_list=[],
    das_size=2.0,
    das_edge_size=0.0,
    xlim=(-1.1, 1.1),
    ylim=(-1.1, 1.1),
):
    """## plot beachball only
    2024/12/05
    """
    df_gafocal, df_hout = _preprocess_focal_files(
        gafocal_txt=gafocal_txt, polarity_dout=polarity_dout
    )
    if len(event_list) == 0:
        event_list = range(len(df_gafocal))

    with mp.Pool(processes=processes) as pool:
        pool.starmap(
            get_tracer_beach,
            [
                (
                    df_gafocal,
                    polarity_dout,
                    df_hout,
                    event_index,
                    output_dir,
                    gamma_events,
                    gamma_detailed,
                    get_waveform,
                    sac_parent_dir,
                    h5_parent_dir,
                    input_tracer,
                    tt_table,
                    extend_pol_picks,
                    station,
                    das_size,
                    das_edge_size,
                    xlim,
                    ylim,
                )
                for event_index in event_list
            ],
        )


# plot pol map according to gafocal result
def plot_polarity_profiles(
    station: Path,
    gamma_events: Path,
    h3dd_events: Path,
    polarity_picks: Path,
    polarity_dout: Path,
    focal_catalog: Path,
    figure_dir: Path,
    event_index: int,
    region: list | None = None,
    depth_lim: tuple | None = None,
    focal_xlim=(-1.1, 1.1),
    focal_ylim=(-1.1, 1.1),
    focal_add_info=False,
    focal_only_das=False,
    savefig=True,
):
    """## Plot the polarity information on the station, typically for DAS.

    Args:
        Event_index: event_index here is the row num of gafocal.
        gamma_events: reorder one due to contain the column of h3dd_index.
    ### Usage
    no need to think too much, just follow the hint and input your Path.
    #NOTE: 2024/12/2 using focal row to plot polarity_map.
    """
    figure_dir.mkdir(parents=True, exist_ok=True)

    df_gafocal, df_hout = _preprocess_focal_files(
        gafocal_txt=focal_catalog, polarity_dout=polarity_dout
    )

    focal_dict, h3dd_index = find_gafocal_polarity(
        df_gafocal=df_gafocal,
        df_hout=df_hout,
        polarity_dout=polarity_dout,
        event_index=event_index,
        get_h3dd_index=True,
    )
    # available_sta = list(focal_dict['sta_info'].keys())
    gamma_index = find_gamma_index(gamma_events=gamma_events, h3dd_index=h3dd_index)
    # data preprocessing
    df_h3dd, _ = check_format(catalog=h3dd_events)
    df_h3dd = df_h3dd.loc[h3dd_index]
    df_station = pd.read_csv(station)
    if region is None:
        region = [
            df_station['longitude'].min(),
            df_station['longitude'].max(),
            df_station['latitude'].min(),
            df_station['latitude'].max(),
        ]

    df_polarity_picks = pd.read_csv(polarity_picks)
    df_polarity_picks = df_polarity_picks[
        df_polarity_picks['event_index'] == gamma_index
    ]
    df_polarity_picks = df_polarity_picks.rename(columns={'station_id': 'station'})

    # The frame of plot
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 14),
        gridspec_kw={'width_ratios': [2, 1], 'height_ratios': [2, 1]},
    )

    geo_ax = fig.add_subplot(2, 2, 1, projection=ccrs.PlateCarree())
    geo_ax1 = fig.add_subplot(4, 4, 1, projection=ccrs.PlateCarree())

    # Main map
    get_polarity_map(
        geo_ax=geo_ax,
        df_station=df_station,
        df_polarity_picks=df_polarity_picks,
        region=region,
        title=f"{focal_dict['utc_time']}\nLat: {focal_dict['latitude']}, Lon: {focal_dict['longitude']}, Depth: {focal_dict['depth']}",
        epi_lon=df_h3dd['longitude'].iloc[0],
        epi_lat=df_h3dd['latitude'].iloc[0],
    )

    # Adding sub map top left
    _add_sub_map(
        geo_ax1,
        epi_lon=df_h3dd['longitude'].iloc[0],
        epi_lat=df_h3dd['latitude'].iloc[0],
    )  # 121.58, 23.86)
    plot_station(df_station, geo_ax1)

    xlim = geo_ax.get_xlim()
    ylim = geo_ax.get_ylim()

    get_profile(
        df_station=df_station,
        ax=axes[0, 1],
        df_polarity_picks=df_polarity_picks,
        ylim=ylim,
        depth_axis='x',
        depth_lim=depth_lim,
    )

    get_profile(
        df_station=df_station,
        ax=axes[1, 0],
        df_polarity_picks=df_polarity_picks,
        xlim=xlim,
        depth_axis='y',
        depth_lim=depth_lim,
    )

    get_beach(
        focal_dict=focal_dict,
        ax=axes[1, 1],
        xlim=focal_xlim,
        ylim=focal_ylim,
        only_das=focal_only_das,
        add_info=focal_add_info,
    )
    axes[0, 0].axis('off')

    # plt.tight_layout()
    if savefig:
        plt.savefig(
            figure_dir
            / f"Event_{event_index}_{focal_dict['utc_time'].replace('-','_').replace(':', '_')}_map.png",
            dpi=300,
            bbox_inches='tight',
        )  # using fig.savefig when we have several figs.


def get_mag_distribution(df, name, start_date, end_date, fig_dir):
    color_map = {
        0: 'green',
        1: 'red',
        2: 'blue',
        3: 'green',
        4: 'orange',
        5: 'blue',
        6: 'red',
    }
    label_map = {
        0: 'seis 6P2S\n(only using seismometer)',
        1: 'seis 6P2S + DAS 15P',
        2: 'seis 6P2S',
        3: 'DAS 15P',
        4: 'Not reach the standard',
        5: 'GDMS Catalog',
        6: 'Seis + DAS',
    }
    df['datetime'] = pd.to_datetime(df['time'])
    df = df[df['magnitude'] > 0]
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(df['magnitude'], bins=16, range=(0, 8), edgecolor='k')
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
        fig_dir / f'{name}_mag_histogram.png',
        dpi=300,
        bbox_inches='tight',
    )

    # Show correlation with scatter plot
    plt.figure(figsize=(15, 10))
    for event_type, group in df.groupby('event_type'):
        plt.scatter(
            group['datetime'],
            group['magnitude'],
            alpha=0.7,
            color=color_map[event_type],
            s=5,
            label=f'{label_map[event_type]}: {len(group)}',
        )
    plt.title('Magnitude vs Date', fontsize=25, pad=10)
    plt.xlabel('Date', fontsize=25, labelpad=10)
    plt.ylabel('Magnitude', fontsize=25, labelpad=10)
    plt.grid(True)
    plt.legend(loc='upper right', labelspacing=1.5, fontsize=15)
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
        fig_dir / f'{name}_mag_daily.png',
        dpi=300,
        bbox_inches='tight',
    )


def get_mags_dist(
        df_list: list[pd.DataFrame], start_date: str, end_date: str, fig_dir, color_list: list, name_list: list, save_name=''
        ):
    """
    Date format: '2024-04-01'
    """
    color_map = {
        0: 'green',
        1: 'red',
        2: 'blue',
        3: 'green',
        4: 'orange',
        5: 'blue',
        6: 'red',
    }
    label_map = {
        0: 'seis 6P2S\n(only using seismometer)',
        1: 'seis 6P2S + DAS 15P',
        2: 'seis 6P2S',
        3: 'DAS 15P',
        4: 'Not reach the standard',
        5: 'GDMS Catalog',
        6: 'Seis + DAS',
    }
    plt.figure(figsize=(15, 10))
    for df, color, name in zip(df_list, color_list, name_list):
        df['datetime'] = pd.to_datetime(df['time'])
        df = df[df['magnitude'] > 0]
        # Show correlation with scatter plot
        if 'event_type' in df.columns:
            for event_type, group in df.groupby('event_type'):
                plt.scatter(
                    group['datetime'],
                    group['magnitude'],
                    alpha=0.7,
                    color=color_map[event_type],
                    s=5,
                    label=f'{label_map[event_type]}: {len(group)}',
                )
        else:
            plt.scatter(
                df['datetime'],
                df['magnitude'],
                alpha=0.7,
                color=color,
                s=5,
                label=f'{name} event num: {len(df)}',
            )            
    plt.title('Magnitude temporal variation', fontsize=25, pad=10)
    plt.xlabel('Date', fontsize=25, labelpad=10)
    plt.ylabel('Magnitude', fontsize=25, labelpad=10)
    plt.grid(True)
    plt.legend(loc='upper right', markerscale=2, labelspacing=1.5, fontsize=15)
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
    ax.set_xlim([pd.Timestamp(start_date), pd.Timestamp(end_date)])
    ax.set_ylim(0, 8)
    plt.savefig(
        fig_dir / f'mag_dist_{save_name}.png',
        dpi=300,
        bbox_inches='tight',
    )


def gamma_h3dd_combine(gamma_events: Path, h3dd_hout: Path, output: Path):
    # reorder
    df_events = pd.read_csv(gamma_events)
    df_events['sort_time'] = pd.to_datetime(df_events['time'])
    df_events = df_events.sort_values(by='sort_time')
    df_events.drop(columns=['sort_time'], inplace=True)
    df_events['h3dd_index'] = df_events.index

    # adding h3dd
    df_h3dd, _ = check_format(h3dd_hout)
    df_h3dd = df_h3dd.add_prefix('h3dd_')
    df_h3dd['h3dd_index'] = df_h3dd.index

    # merge
    df = pd.merge(df_events, df_h3dd, on='h3dd_index', how='left')
    df.to_csv(output, index=False)
    return df

def calculate_fault_azimuth(fault_start, fault_end):
    """
    計算斷層的方向角（以正北為基準）。
    """
    geod = Geod(ellps="WGS84")
    lon1, lat1 = fault_start
    lon2, lat2 = fault_end
    fault_azimuth, _, _ = geod.inv(lon1, lat1, lon2, lat2)
    return float(fault_azimuth)

def project_events_to_profile(df_catalog: pd.DataFrame, length=10, width=80, depth_range=(0, 60),
                              center: tuple | None = None, azimuth: float | None = None,
                              fault_start: tuple | None = None, fault_end: tuple | None = None, temporal=False
                              ):
    """
    Return the df in the profile range.
    """
    geod = Geod(ellps="WGS84")

    # 計算斷層中點
    if azimuth is None and fault_start is not None and fault_end is not None:
        lon1, lat1 = fault_start
        lon2, lat2 = fault_end
        azimuth = calculate_fault_azimuth(fault_start, fault_end)
        mid_lon, mid_lat = (lon1 + lon2) / 2, (lat1 + lat2) / 2
    elif azimuth is not None and center is not None:
        mid_lon, mid_lat = center
    # 投影地震事件到剖面方向
    results = []
    for _, event in df_catalog.iterrows():
        event_lon, event_lat, event_depth = event['longitude'], event['latitude'], event['depth_km']
        if temporal:
            day = event['day']
        # 計算事件的方位角和距離
        event_az, _, event_dist = geod.inv(mid_lon, mid_lat, event_lon, event_lat)

        # 計算事件在剖面方向與垂直方向的距離分量
        relative_angle = np.radians(event_az - azimuth)  # 方位角差值
        y = event_dist * np.cos(relative_angle) / 1000  # 剖面方向距離（公里）
        x = event_dist * np.sin(relative_angle) / 1000  # 垂直剖面距離（公里）

        # 篩選符合剖面範圍的事件
        if (
            abs(y) <= length / 2 and  # 剖面長度範圍
            abs(x) <= width / 2 and  # 剖面寬度範圍
            depth_range[0] <= event_depth <= depth_range[1]  # 深度範圍
        ):
            if temporal:
                results.append((day, x+width/2, y, event_lon, event_lat, event_depth))
            else:
                results.append((x+width/2, y, event_lon, event_lat, event_depth))
    # 返回結果 DataFrame
    if temporal:
        results_df = pd.DataFrame(results, columns=['day', 'x', 'y', 'longitude', 'latitude', 'depth'])
    else:
        results_df = pd.DataFrame(results, columns=['x', 'y', 'longitude', 'latitude', 'depth'])
    return results_df

def add_profile_rectangle(geo_ax, length=10, width=80, center: tuple | None = None, azimuth: float | None = None,
                              fault_start: tuple | None = None, fault_end: tuple | None = None, **kwargs):
    """
    Draws a dashed rectangle representing the profile range on a Cartopy GeoAxes.

    Parameters:
        geo_ax (GeoAxes): Cartopy GeoAxes object.
        fault_start (tuple): Starting point of the fault (longitude, latitude).
        fault_end (tuple): Ending point of the fault (longitude, latitude).
        length (float): Profile length in kilometers.
        width (float): Profile width in kilometers.
        fault_azimuth (float): Fault direction angle (degrees), measured clockwise from North.
        **kwargs: Additional plotting arguments for matplotlib (e.g., color, linestyle).

    Returns:
        None
    """
    geod = Geod(ellps="WGS84")

    # Calculate the midpoint of the profile
    if fault_start is not None and fault_end is not None:
        mid_lon, mid_lat = (fault_start[0] + fault_end[0]) / 2, (fault_start[1] + fault_end[1]) / 2
        azimuth = calculate_fault_azimuth(fault_start, fault_end)
    elif center is not None and azimuth is not None:
        mid_lon , mid_lat = center
    else:
        raise IndexError('Please provide the parameters')
    # Calculate half-length and half-width
    half_length = length / 2
    half_width = width / 2

    # Step 1: Extend the midpoint to get the front and back center points
    front_center_lon, front_center_lat, _ = geod.fwd(mid_lon, mid_lat, azimuth, half_length * 1000)
    back_center_lon, back_center_lat, _ = geod.fwd(mid_lon, mid_lat, azimuth, -half_length * 1000)

    # Step 2: From the front and back center points, extend to the left and right to get the corners
    corners = []

    for center_lon, center_lat in [(front_center_lon, front_center_lat), (back_center_lon, back_center_lat)]:
        # Left side (-90 degrees from fault direction)
        left_lon, left_lat, _ = geod.fwd(center_lon, center_lat, azimuth - 90, half_width * 1000)
        # Right side (+90 degrees from fault direction)
        right_lon, right_lat, _ = geod.fwd(center_lon, center_lat, azimuth + 90, half_width * 1000)
        corners.append((left_lon, left_lat))
        corners.append((right_lon, right_lat))

    # Step 3: Reorder the corners to form a closed rectangle (clockwise)
    corners = [corners[0], corners[2], corners[3], corners[1], corners[0]]

    # Plot the rectangle on the GeoAxes
    lons, lats = zip(*corners)
    geo_ax.plot(lons, lats, transform=ccrs.PlateCarree(), **kwargs)

def add_profile_line(
        geo_ax: Axes,
        extend_km: Any,
        azimuth: Any,
        start_point: tuple | None = None,
        fault_start: tuple | None = None,
        fault_end: tuple | None = None,
        **kwargs
        ):
    """
    Draws a dashed rectangle representing the profile range on a Cartopy GeoAxes.

    Parameters:
        geo_ax (GeoAxes): Cartopy GeoAxes object.
        fault_start (tuple): Starting point of the fault (longitude, latitude).
        fault_end (tuple): Ending point of the fault (longitude, latitude).
        length (float): Profile length in kilometers.
        width (float): Profile width in kilometers.
        fault_azimuth (float): Fault direction angle (degrees), measured clockwise from North.
        **kwargs: Additional plotting arguments for matplotlib (e.g., color, linestyle).

    Returns:
        None
    """
    geod = Geod(ellps="WGS84")

    # Calculate the midpoint of the profile
    if fault_start is not None and fault_end is not None:
        mid_lon, mid_lat = (fault_start[0] + fault_end[0]) / 2, (fault_start[1] + fault_end[1]) / 2
        azimuth = calculate_fault_azimuth(fault_start, fault_end) # parallel to fault
        start_lon, start_lat, _ = geod.fwd(mid_lon, mid_lat, azimuth - 90, (extend_km / 2)*1000)
        azimuth = azimuth + 90 # perpendicular to fault
    elif start_point is not None and azimuth is not None:
        start_lon , start_lat = start_point
    else:
        raise IndexError('Please provide the parameters')

    # the extend point
    end_lon, end_lat, _ = geod.fwd(start_lon, start_lat, azimuth, extend_km * 1000)

    # Step 2: From the front and back center points, extend to the left and right to get the corners
    corners = []
    corners.append((start_lon, start_lat))
    corners.append((end_lon, end_lat))

    # Plot the rectangle on the GeoAxes
    lons, lats = zip(*corners)
    geo_ax.plot(lons, lats, transform=ccrs.PlateCarree(), **kwargs)

def add_single_profile(events_df, ax: Axes, width: Any, depth_range=(0, 60), temporal=True,
                       manual_min=None, manual_max=None, **kwargs):
    """
    Plots a seismic profile on a given axis.
    
    Parameters:
        events_df (pd.DataFrame): Contains 'x' (distance along profile) and 'depth' columns.
        ax (matplotlib.axes.Axes): The axis to plot on.
        width (float): Profile width (in km).
        depth_range (tuple): Depth range (min_depth, max_depth) in km.
        profile (str): Profile name (for labeling).
    """
    if temporal:
        colormap = plt.cm.viridis
        norm = mcolors.Normalize(
            vmin=events_df['day'].min(),
            vmax=events_df['day'].max()
            )  # Normalize days to range [0, 1]
        if manual_min is not None and manual_max is not None:
            norm = mcolors.Normalize(
                vmin=manual_min,
                vmax=manual_max
                )             
        for day, group in events_df.groupby('day'):
            # print(f'day {day}: {len(group)}')
            color = colormap(norm(day))    
            ax.scatter(
                group['x'],
                group['depth'],
                c='k',#color,
                s=5,
                alpha=0.8
                )
    else:
        ax.scatter(
            events_df['x'],
            events_df['depth'],
            s=5,
            **kwargs
            )
    # ax.tick_params(axis="x", which="both", bottom=True, labelbottom=True)
    ax.set_xlim(0, width)#(-width / 2, width / 2)
    ax.set_ylim(depth_range[0], depth_range[1])
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Depth (km)")
    ax.invert_yaxis()
    # ax.grid()

def _daily_job(figsize, num_columns, profile_nums, start_point, other_catalog, length, width, depth_range, group, 
              num_rows, azimuth, topo_csv, plot_vel, vel_df, vel_mode, vel_cmap, manual_min, manual_max, save_dir, day):
    fig = plt.figure(figsize=figsize)
    outer_gs = GridSpec(num_rows, num_columns, figure=fig)   
    letters = string.ascii_uppercase
    for i, (dis, letter) in enumerate(zip(np.linspace(0, (length / 2)*(profile_nums-1), profile_nums), letters[0:profile_nums])):    
        geod = Geod(ellps='WGS84')
        next_lon, next_lat, _ = geod.fwd(start_point[0], start_point[1], azimuth, dis*1000)
        print(f'Next start point: ({next_lon}, {next_lat})')
        new_lon, new_lat, _ = geod.fwd(next_lon, next_lat, azimuth + 90, (width / 2)*1000)
        new_center = (new_lon, new_lat)
        # plot background events
        if other_catalog is not None:
            other_events = project_events_to_profile(
                df_catalog=other_catalog,
                length=length, 
                width=width,
                depth_range=depth_range,
                center=new_center,
                azimuth=20,
                )                
        profile_events = project_events_to_profile(
            df_catalog=group,
            length=length, 
            width=width,
            depth_range=depth_range,
            center=new_center,
            azimuth=20
            )
        row = i % num_rows
        col = i // num_rows
        inner_gs = GridSpecFromSubplotSpec(
            2,
            3,
            subplot_spec=outer_gs[row, col],
            height_ratios=[1, 3],
            width_ratios=[1, 1, 0.05]
        )
        # plotting topo
        ax1 = fig.add_subplot(inner_gs[0, :-1])
        distances, elevations = extract_topo_profile_with_geod(
            csv_path=topo_csv,
            start_point=(next_lon, next_lat),
            azimuth=azimuth+90,
            distance_km=width
            )
        add_topo_profile(
            ax=ax1,
            distances=distances,
            elevations=elevations,
            profile_letter=letter
        )
        # plotting profile
        ax2 = fig.add_subplot(inner_gs[1, :-1])
        cax = fig.add_subplot(inner_gs[1, -1:])
        if plot_vel:

            #TODO: Adding a condition to solve possibly unbound.
            add_vel_profile(
                ax=ax2,
                vel_df=vel_df,
                start_point=(next_lon, next_lat),
                mode=vel_mode,
                cax=cax,
                cmap=vel_cmap,
                distance=width,
                azimuth=azimuth
            )
        if other_catalog is not None:
            add_single_profile(
                other_events,
                ax2,
                temporal=False,
                c='gray',
                alpha=0.5,
                width=width,
                depth_range=depth_range
                )                
        add_single_profile(
            profile_events,
            ax2,
            manual_min=manual_min,
            manual_max=manual_max,
            width=width,
            depth_range=depth_range
            )
    plt.tight_layout()
    plt.savefig(save_dir / f'gridspec_cmap_{day}.png', dpi=300)
    plt.close()

def daily_profile(
        df_catalog: pd.DataFrame,
        topo_csv,
        start_point=(121.08, 23.65),
        other_catalog: pd.DataFrame | None = None,
        length=10,
        width=80,
        azimuth=20,
        depth_range=(0, 60),
        save_dir=Path(''),
        save_name='',
        profile_nums=21,
        num_rows=3,
        figsize=(),
        manual_min=1,
        manual_max=17,
        plot_vel=False,
        vel_txt: Path | None = None,
        vel_mode='vpt',
        vel_cmap='RdYlBu',
        temporal=False
        ):
    """
    Plots a seismic profile on a given axis.
    
    Parameters:
        azimuth (int | float): the profile moving azimuth, to be more specific, the slope between the start point of A-A' and B-B'
    """
    if profile_nums % num_rows == 0:
        num_columns = profile_nums // num_rows
    else:    
        num_columns = (profile_nums // num_rows) + 1
    if len(figsize) == 0:
        figsize=(num_columns * 10, num_rows * 7)

    geod = Geod(ellps='WGS84')
    
    if vel_txt is not None:
        vel_df = pd.read_csv(vel_txt, sep='\s+')

    if temporal:
        args =[
            (figsize, num_columns, profile_nums, start_point,
            other_catalog, length, width, depth_range, group, 
            num_rows, azimuth, topo_csv, plot_vel, vel_df, vel_mode,
            vel_cmap, manual_min, manual_max, save_dir, day)
            for day, group in df_catalog.groupby('day')
            ]        
        with mp.Pool(processes=20) as pool:
            pool.starmap(_daily_job, args)
    else:
        fig = plt.figure(figsize=figsize)
        outer_gs = GridSpec(num_rows, num_columns, figure=fig)   
        letters = string.ascii_uppercase
        for i, (dis, letter) in enumerate(
            zip(
                np.linspace(0, (length / 2)*(profile_nums-1), profile_nums),
                letters[0:profile_nums]
                )
                ):
            geod = Geod(ellps='WGS84')
            next_lon, next_lat, _ = geod.fwd(start_point[0], start_point[1], azimuth, dis*1000)
            print(f'Next start point: ({next_lon}, {next_lat})')
            new_lon, new_lat, _ = geod.fwd(next_lon, next_lat, azimuth + 90, (width / 2)*1000)
            new_center = (new_lon, new_lat)
            print(f'Center point: {new_center}')
            # plot background events
            if other_catalog is not None:
                other_events = project_events_to_profile(
                    df_catalog=other_catalog,
                    length=length, 
                    width=width,
                    depth_range=depth_range,
                    center=new_center,
                    azimuth=azimuth,
                    )                
            profile_events = project_events_to_profile(
                df_catalog=df_catalog,
                length=length, 
                width=width,
                depth_range=depth_range,
                center=new_center,
                azimuth=azimuth
                )
            row = i % num_rows
            col = i // num_rows
            inner_gs = GridSpecFromSubplotSpec(
                2,
                3,
                subplot_spec=outer_gs[row, col],
                height_ratios=[1, 3],
                width_ratios=[1, 1, 0.05]
            )
            # plotting topo
            ax1 = fig.add_subplot(inner_gs[0, :-1])
            distances, elevations = extract_topo_profile_with_geod(
                csv_path=topo_csv,
                start_point=(next_lon, next_lat),
                azimuth=azimuth+90,
                distance_km=width
                )
            add_topo_profile(
                ax=ax1,
                distances=distances,
                elevations=elevations,
                profile_letter=letter
            )
            # plotting profile
            ax2 = fig.add_subplot(inner_gs[1, :-1])
            cax = fig.add_subplot(inner_gs[1, -1:])
            if plot_vel:
                #TODO: Adding a condition to solve possibly unbound.
                add_vel_profile(
                    ax=ax2,
                    vel_df=vel_df,
                    start_point=(next_lon, next_lat),
                    mode=vel_mode,
                    cax=cax,
                    cmap=vel_cmap,
                    distance=width,
                    azimuth=azimuth + 90
                )
            if other_catalog is not None:
                add_single_profile(
                    other_events,
                    ax2,
                    temporal=False,
                    c='gray',
                    alpha=0.5,
                    width=width,
                    depth_range=depth_range
                    )                
            add_single_profile(
                profile_events,
                ax2,
                # manual_min=manual_min,
                # manual_max=manual_max,
                temporal=False,
                c='k',
                width=width,
                depth_range=depth_range
                )
        plt.tight_layout()
        plt.savefig(save_dir / save_name, dpi=300)
        plt.close()


def mapview_with_profile(
    station: Path,
    topo_csv,
    catalog: pd.DataFrame,
    other_catalog: pd.DataFrame | None = None,
    depth_range=(0, 60),
    profile_nums=6,
    num_rows=3,
    catalog_filter_: dict | None = None,
    title='Event distribution',
    eq_list=[],
    main_eq_size=10,
    plot_station_name=True,
    savefig=True,
    fig_dir: Path | None = None,
    h3dd_mode=False,
    prefix='',
    savename='',
    temporal=False,
    plot_profile_line=False,
    horizontal_start_point=(121.08, 23.65),
    horizontal_movement={'length': 10, 'width': 80, 'azimuth': 20, 'steps': np.linspace(0, 100, 21), 'letters': Any}, # length, width, azimuth
    vertical_start_point=(121.485, 24.667),
    vertical_movement={'length': 10, 'width': 120, 'azimuth': 110, 'steps': np.linspace(0, 70, 15), 'letters': Any},
    target_start_point=(121.485, 24.667),
    target_movement={'length': 10, 'width': 120, 'azimuth': 110, 'steps': np.linspace(0, 70, 15), 'letters': Any},
    plot_vel_on_mapview=False,
    vel_txt: Path | None = None,
    vel_mode='vpt',
    vel_cmap='RdYlBu'
):
    """
    catalog should at least contains the lon and lat.

    """
    # plot setting
    fig = plt.figure(figsize=(32, 18))
    parent_gs = GridSpec(1, 2, width_ratios=[1, 1])

    geo_ax = fig.add_subplot(parent_gs[0, 0], projection=ccrs.PlateCarree())

    if catalog_filter_ is None:
        region = [
            catalog['longitude'].min() - 0.2,
            catalog['longitude'].max() + 0.2,
            catalog['latitude'].min() - 0.2,
            catalog['latitude'].max() + 0.2,
        ]
    else:
        region = [
            catalog_filter_['min_lon'],
            catalog_filter_['max_lon'],
            catalog_filter_['min_lat'],
            catalog_filter_['max_lat'],
        ]

    get_mapview(region=region, ax=geo_ax, title=title, fontsize=25)
    if plot_vel_on_mapview and vel_txt is not None:
        vel_df = pd.read_csv(vel_txt, sep='\s+')
        add_vel_view(
            ax=geo_ax,
            vel_df=vel_df,
            region=region
        )
    # ax setting
    geo_ax.set_xlim(region[0], region[1])
    geo_ax.set_ylim(region[2], region[3])
    # trim the catalog
    catalog = catalog_filter(
        catalog_df=catalog, h3dd_mode=h3dd_mode, catalog_range=catalog_filter_
    )
    # get the size corresponding to magnitude
    catalog.loc[:, 'size'] = catalog['magnitude'].apply(_get_size)
    # station
    df_station = pd.read_csv(station)
    # print(
    #     'Geo Axes Patch Extent:',
    #     geo_ax.get_window_extent(renderer=fig.canvas.get_renderer()),
    # )
    plot_station(
        df_station=df_station,
        geo_ax=geo_ax,
        region=region,
        plot_station_name=plot_station_name,
        text_dist=0.005,
    )
    manual_elements = [
        Line2D(
            [0],
            [0],
            marker='^',
            color='w',
            label='Seismometer',
            markersize=7,
            markerfacecolor='c',
            markeredgecolor='k',
        ),
    ]
    # add all events
    event_num = len(catalog)
    manual_elements.append(
        Line2D(
            [0],
            [0],
            marker='o',
            color='w',
            label=f'Event num: {event_num}',
            markersize=5,
            markerfacecolor='k',
            markeredgecolor='k',
        )
    )       
    if temporal:
        colormap = plt.cm.viridis
        norm = mcolors.Normalize(
            vmin=catalog['day'].min(),
            vmax=catalog['day'].max()
            )  # Normalize days to range [0, 1]
        for day, group in catalog.groupby('day'):
            print(f'day {day}: {len(group)}')
            color = colormap(norm(day))
            geo_ax.scatter(
                group[f'{prefix}longitude'],
                group[f'{prefix}latitude'],
                s=group['size'],
                c=color,
                alpha=0.5,
                label=f'Day {day} event num: {len(group)}',
                zorder=3,
            )
            # Add a colorbar to indicate progression
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        # Create an inset for the colorbar within geo_ax
        cbar_ax = inset_axes(
            geo_ax,  # Parent axis (geo_ax)
            width="3%",  # Width of the colorbar relative to geo_ax
            height="50%",  # Height of the colorbar relative to geo_ax
            loc="lower right",  # Location inside the geo_ax
            bbox_to_anchor=(0.05, 0.05, 0.85, 0.9),  # Adjust position within geo_ax
            bbox_transform=geo_ax.transAxes,  # Use geo_ax coordinates
            borderpad=0,  # Padding around the inset
        )
        cbar = plt.colorbar(sm, cax=cbar_ax, orientation='vertical')
        cbar.set_label("Day Since 0401")
    else:
        geo_ax.scatter(
            catalog[f'{prefix}longitude'],
            catalog[f'{prefix}latitude'],
            s=catalog['size'],
            c='k',
            alpha=0.5,
            zorder=3,
        )         

    # profile
    geod = Geod(ellps='WGS84')
    if plot_profile_line:
        # letters = string.ascii_uppercase
        # horizontal
        for i, (dis, letter) in enumerate(zip(horizontal_movement['steps'], horizontal_movement['letters'])):
            next_lon, next_lat, _ = geod.fwd(horizontal_start_point[0], horizontal_start_point[1], horizontal_movement['azimuth'], dis*1000)
            text_lon, text_lat, _ = geod.fwd(horizontal_start_point[0], horizontal_start_point[1], horizontal_movement['azimuth'], dis*1000)
            add_profile_line(geo_ax=geo_ax, extend_km=horizontal_movement['width'], start_point=(next_lon, next_lat), azimuth=horizontal_movement['azimuth'] + 90, alpha=0.7, c='k')   
            geo_ax.text(x=text_lon -0.02, y=text_lat, s=f"{letter}-{letter}'")
        
        # vertical
        for i, (dis, letter) in enumerate(zip(vertical_movement['steps'], vertical_movement['letters'])):
            next_lon, next_lat, _ = geod.fwd(vertical_start_point[0], vertical_start_point[1], vertical_movement['azimuth'], dis*1000)
            text_lon, text_lat, _ = geod.fwd(vertical_start_point[0], vertical_start_point[1], vertical_movement['azimuth'], dis*1000)
            add_profile_line(geo_ax=geo_ax, extend_km=vertical_movement['width'], start_point=(next_lon, next_lat), azimuth=vertical_movement['azimuth'] + 90, alpha=0.7, c='k')   
            geo_ax.text(x=text_lon -0.02, y=text_lat, s=f"{letter}-{letter}'")                    

    # add main eq
    for eq in eq_list:
        _add_epicenter(
            x=eq[0], y=eq[1], ax=geo_ax, size=main_eq_size, alpha=0.7, color=eq[4]
        )
              
        manual_elements.append(
            Line2D(
                [0],
                [0],
                marker='*',
                color='w',
                label=eq[3],
                markersize=10,
                markerfacecolor=eq[4],
                markeredgecolor='k',
            )
        )
    # geo_ax.autoscale(tight=True)
    geo_ax.set_aspect('auto')
    geo_ax.legend(
        handles=manual_elements,  # + legend_elements,
        loc='upper left',
        markerscale=2,
        labelspacing=1.5,
        borderpad=1,
        fontsize=10,
        framealpha=0.6,
    )

    # plotting profiles
    if profile_nums % num_rows == 0:
        num_columns = profile_nums // num_rows
    else:    
        num_columns = (profile_nums // num_rows) + 1

    if vel_txt is not None:
        vel_df = pd.read_csv(vel_txt, sep='\s+')
    outer_gs = GridSpecFromSubplotSpec(
        num_rows,
        num_columns,
        subplot_spec=parent_gs[0, 1],
    )
    for i, (dis, letter) in enumerate(zip(target_movement['steps'], target_movement['letters'])):
        geod = Geod(ellps='WGS84')
        next_lon, next_lat, _ = geod.fwd(target_start_point[0], target_start_point[1], target_movement['azimuth'], dis*1000)
        print(f'Next start point: ({next_lon}, {next_lat})')
        add_profile_line(
            geo_ax=geo_ax,
            extend_km=target_movement['width'],
            azimuth=target_movement['azimuth'] + 90,
            start_point=(next_lon, next_lat),
            c='yellow'
        )
        new_lon, new_lat, _ = geod.fwd(next_lon, next_lat, target_movement['azimuth'] + 90, (target_movement['width'] / 2)*1000)
        new_center = (new_lon, new_lat)
        print(f'Center point: {new_center}')
        # plot background events
        if other_catalog is not None:
            other_events = project_events_to_profile(
                df_catalog=other_catalog,
                length=target_movement['length'], 
                width=target_movement['width'],
                depth_range=depth_range,
                center=new_center,
                azimuth=target_movement['azimuth'],
                )                
        profile_events = project_events_to_profile(
            df_catalog=catalog,
            length=target_movement['length'], 
            width=target_movement['width'],
            depth_range=depth_range,
            center=new_center,
            azimuth=target_movement['azimuth']
            )
        row = i % num_rows
        col = i // num_rows
        inner_gs = GridSpecFromSubplotSpec(
            2,
            3,
            subplot_spec=outer_gs[row, col],
            height_ratios=[1, 3],
            width_ratios=[1, 1, 0.05]
        )
        # plotting topo
        ax1 = fig.add_subplot(inner_gs[0, :-1])
        distances, elevations = extract_topo_profile_with_geod(
            csv_path=topo_csv,
            start_point=(next_lon, next_lat),
            azimuth=target_movement['azimuth']+90,
            distance_km=target_movement['width']
            )
        add_topo_profile(
            ax=ax1,
            distances=distances,
            elevations=elevations,
            profile_letter=letter
        )
        # plotting profile
        ax2 = fig.add_subplot(inner_gs[1, :-1])
        cax = fig.add_subplot(inner_gs[1, -1:])
        if vel_txt is not None:
            #TODO: Adding a condition to solve possibly unbound.
            add_vel_profile(
                ax=ax2,
                vel_df=vel_df,
                start_point=(next_lon, next_lat),
                mode=vel_mode,
                cax=cax,
                cmap=vel_cmap,
                distance=target_movement['width'],
                azimuth=target_movement['azimuth'] + 90
            )
        if other_catalog is not None:
            add_single_profile(
                other_events,
                ax2,
                temporal=False,
                c='gray',
                alpha=0.5,
                width=target_movement['width'],
                depth_range=depth_range
                )                
        add_single_profile(
            profile_events,
            ax2,
            # manual_min=manual_min,
            # manual_max=manual_max,
            temporal=False,
            c='k',
            width=target_movement['width'],
            depth_range=depth_range
            )    
    
    if savefig and fig_dir is not None:
        plt.tight_layout()
        plt.savefig(fig_dir / f'{savename}.png', dpi=300, bbox_inches='tight')