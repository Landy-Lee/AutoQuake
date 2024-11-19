# import os
# import glob
# import math

# import calendar
# from datetime import datetime, timedelta
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

# from matplotlib.ticker import MultipleLocator
import pandas as pd
from cartopy.mpl.geoaxes import GeoAxes

# from obspy.imaging.beachball import beach
# import multiprocessing as mp
# from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
# from collections import defaultdict
# from pyrocko import moment_tensor as pmt
# from pyrocko.plot import beachball, mpl_color
from ._plot_base import *
from .comp_plot import find_compared_gafocal_polarity


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
    # map_proj = ccrs.PlateCarree()
    # tick_proj = ccrs.PlateCarree()
    # if main_ax is None and sub_ax is None:
    #     fig = plt.figure(figsize=(8, 12))
    # gs = GridSpec(2, 1, height_ratios=[3, 1])
    df_station = pd.read_csv(station)
    df_seis_station = df_station[df_station['station'].map(station_mask)]
    df_das_station = df_station[~df_station['station'].map(station_mask)]
    if not df_seis_station.empty:
        # get seismic station
        if main_ax is None:
            main_ax = fig.add_subplot(gs[0], projection=map_proj)
        if main_region is None:
            main_region = [
                df_seis_station['longitude'].min() - 0.5,
                df_seis_station['longitude'].max() + 0.5,
                df_seis_station['latitude'].min() - 0.5,
                df_seis_station['latitude'].max() + 0.5,
            ]
        get_mapview(main_region, main_ax)
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
                color='r',
                s=100,
                zorder=4,
                label='GaMMA',
            )
        if use_h3dd and event_dict is not None:
            main_ax.scatter(
                x=event_dict['h3dd']['event_lon'],
                y=event_dict['h3dd']['event_lat'],
                marker='*',
                color='b',
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
        get_mapview(sub_region, sub_ax)
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
                color='r',
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
                color='b',
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
    sac_dir_name=None,
    h5_parent_dir=None,
    station_mask=station_mask,
    seis_region=None,
    das_region=None,
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
    else:
        status = 'gamma'
        use_h3dd = False

    # figure setting
    fig = plt.figure()
    map_proj = ccrs.PlateCarree()
    # tick_proj = ccrs.PlateCarree()

    # retrieving data
    if sac_parent_dir is not None and sac_dir_name is not None:
        sac_dict = find_sac_data(
            event_time=event_dict[status]['event_time'],
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
        das_waveform_ax = fig.add_axes([0.82, -0.15, 0.8, 0.55], sharey=das_map_ax)
        if seis_map_ax is None:
            das_waveform_ax.set_title(
                f"Event_{event_i}: {event_dict[status]['event_time']}, lat: {event_dict[status]['event_lat']}, lon: {event_dict[status]['event_lon']}, depth: {event_dict[status]['event_depth']} km"
            )
    else:
        das_map_ax = None
        das_waveform_ax = None
    df_seis_phasenet_picks, df_das_phasenet_picks = find_phasenet_pick(
        event_total_seconds=event_dict[status]['event_total_seconds'],
        sac_dict=sac_dict,
        df_all_picks=df_phasenet_picks,
        station_mask=station_mask,
    )
    df_seis_gamma_picks, df_das_gamma_picks = find_gamma_pick(
        df_gamma_picks=df_event_picks,
        sac_dict=sac_dict,
        event_total_seconds=event_dict[status]['event_total_seconds'],
        station_mask=station_mask,
    )
    plot_waveform_check(
        sac_dict=sac_dict,
        df_seis_phasenet_picks=return_none_if_empty(df_seis_phasenet_picks),
        df_seis_gamma_picks=return_none_if_empty(df_seis_gamma_picks),
        df_das_phasenet_picks=return_none_if_empty(df_das_phasenet_picks),
        df_das_gamma_picks=return_none_if_empty(df_das_gamma_picks),
        event_total_seconds=event_dict[status]['event_total_seconds'],
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
        use_h3dd=use_h3dd,
        station_mask=station_mask,
        main_region=seis_region,
        sub_region=das_region,
    )
    save_name = (
        event_dict[status]['event_time']
        .replace(':', '_')
        .replace('-', '_')
        .replace('.', '_')
    )

    plt.tight_layout()
    plt.savefig(
        fig_dir / f'event_{event_i}_{save_name}.png', bbox_inches='tight', dpi=300
    )
    plt.close()


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

    df_gafocal, _ = check_format(focal_catalog)
    focal_dict = find_compared_gafocal_polarity(
        gafocal_df=df_gafocal,
        polarity_dout=polarity_dout,
        analyze_year='2024',
        event_index=0,  # why this zero?
        use_gamma=False,
        get_waveform=False,
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

    get_das_beach(
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
