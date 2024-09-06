import pygmt
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
from matplotlib.axes import Axes
from matplotlib.colors import LightSource
from cartopy.mpl.geoaxes import GeoAxes
from pathlib import Path
from typing import Any

def h3dd_time_trans(x: str):
    return int(x[:2])*3600 + int(x[2:4])*60 + int(x[4:6]) + float(f"0.{x[6:]}")

def station_mask(df: pd.DataFrame):
    """
    pattern to distinguish the DAS and Seismometer
    """
    return df['station'].str[1].str.isdigit()

def check_format(catalog: dict[int, dict[str, Any]], i):
    if catalog[i]['catalog'].suffix == '.csv':
        df = pd.read_csv(catalog[i]['catalog'])
        timestamp = df["time"].apply(lambda x: datetime.fromisoformat(x).timestamp()).to_numpy()
    else:
        use_cols = [0,1,2,3,4,5]
        col_name = ['ymd', 'hms', 'latitude', 'longitude', 'depth_km', 'mag']
        df = pd.read_csv(catalog[i]['catalog'], sep='\s+', header=None, usecols=use_cols, dtype={1: 'str'}, names=col_name)
        timestamp = df["hms"].apply(h3dd_time_trans).to_numpy()
    return df, timestamp

def status_message_and_fig_name(catalog_list: list[Path], name_list: list[str], use_ori: bool, use_both: bool, use_common: bool, use_main: bool) -> str:
    """
    print the status message and decide the fig name to save
    """
    print(f"Main catalog: {name_list[0]}({catalog_list[0]})\nCompared catalog: {name_list[1]}({catalog_list[1]})\n")
    print("CURRENT STATUS:")
    if use_ori:
        if use_both:
            print(f"Using both catalog to plot the original distribution")
            return f"{name_list[0]}_{name_list[1]}_ori.png"
        else:
            if use_main:
                print(f"Using {name_list[0]} catalog to plot the original distribution")
                return f"{name_list[0]}_ori.png"
            else:
                print(f"Using {name_list[1]} catalog to plot the original distribution")
                return f"{name_list[1]}_ori.png"
    elif use_both:
        if use_common:
            print(f"Using both catalog to plot the common events distribution")
            return f"{name_list[0]}_{name_list[1]}_common.png"
        else:
            print(f"Using both catalog to plot the unique events distribution")
            return f"{name_list[0]}_{name_list[1]}_only.png"
    else:
        if use_common:
            if use_main:
                print(f"Using {name_list[0]} catalog to plot the common events distribution")
                return f"{name_list[0]}_common.png"
            else:
                print(f"Using {name_list[1]} catalog to plot the common events distribution")
                return f"{name_list[1]}_common.png"
        else:
            if use_main:
                print(f"Using {name_list[0]} catalog to plot the unique events distribution")
                return f"{name_list[0]}_only.png"
            else:
                print(f"Using {name_list[1]} catalog to plot the unique events distribution")
                return f"{name_list[1]}_only.png"

def catalog_compare(catalog: dict[int, dict[str, Any]], tol: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_main, timestamp_main = check_format(catalog=catalog, i=0)
    df_comp, timestamp_comp = check_format(catalog=catalog, i=1)
    diff_time = timestamp_main - timestamp_comp[:, np.newaxis]
    main_boolean = (np.abs(diff_time) < tol).any(axis=0) # axis=0 -> comparing the main
    comp_boolean = (np.abs(diff_time) < tol).any(axis=1)
    main_common = df_main[main_boolean]
    comp_common = df_comp[comp_boolean]
    main_only = df_main[~main_boolean]
    comp_only = df_comp[~comp_boolean]
    print(f"Main catalog: {len(main_common)}/{len(df_main)} founded")
    print(f"Compared catalog: {len(comp_common)}/{len(df_comp)} founded")

    return main_common, comp_common, main_only, comp_only

def pack(catalog_list: list[Path], name_list: list[str]) -> dict[int, dict[str, Any]]:
    pack_dict = {}
    for i, (catalog_path, name) in enumerate(zip(catalog_list, name_list)):
        pack_dict[i] = {"catalog": catalog_path, "name": name}
    return pack_dict

def catalog_filter(catalog_df: pd.DataFrame, catalog_range: dict[str, float]) -> pd.DataFrame:
    catalog_df = catalog_df[
            (catalog_df["longitude"] > catalog_range["min_lon"]) 
            & (catalog_df["longitude"] < catalog_range["max_lon"]) 
            & (catalog_df["latitude"] > catalog_range["min_lat"]) 
            & (catalog_df["latitude"] < catalog_range["max_lat"]) 
            & (catalog_df["depth_km"] > catalog_range["min_depth"]) 
            & (catalog_df["depth_km"] < catalog_range["max_depth"])
            ]
    return catalog_df

def plot_ori(catalog: dict[int, dict[str, Any]], catalog_range: dict[str, float], use_both: bool, use_main: bool, axes: Axes, geo_ax: GeoAxes):
    if use_both:
        key_list = catalog.keys()
    elif use_main:
        key_list = [0]
    else:
        key_list = [1]
    for i in key_list:
        if catalog[i]['catalog'].suffix == '.csv':
            catalog_df = pd.read_csv(catalog[i]["catalog"])
        else:
            col_name = ["ymd", "total_seconds", "latitude", "longitude", "depth_km", "par_1", "par_2", "par_3", "par_4", "par_5", "par_51"]
            catalog_df = pd.read_csv(catalog[i]["catalog"], sep='\s+', header=None, names=col_name)
        equip = catalog[i]["name"]

        # cmap = "viridis"
        catalog_df = catalog_filter(
            catalog_df=catalog_df, catalog_range=catalog_range
        )

        geo_ax.scatter(
            catalog_df["longitude"],
            catalog_df["latitude"],
            s=5,
            c="b" if i == 1 else "r",
            alpha=0.5,
            label=f"{equip} event num: {len(catalog_df)}",
            rasterized=True
        )

        axes[0, 1].scatter(
            catalog_df["depth_km"],
            catalog_df["latitude"],
            s=5,
            c="b" if i == 1 else "r",
            alpha=0.5,
            label=f"{equip}",
            rasterized=True,
        )

        axes[1, 0].scatter(
            catalog_df["longitude"],
            catalog_df["depth_km"],
            s=5,
            c="b" if i == 1 else "r",
            alpha=0.5,
            label=f"{equip}",
            rasterized=True,
        )

def plot_bypass(catalog: dict[int, dict[str, Any]], tol: int, use_both: bool, 
                use_main: bool, use_common: bool):
    
    main_common, comp_common, main_only, comp_only = catalog_compare(catalog, tol)
    if use_both:
        if use_common:
            df_list = [main_common, comp_common]
            nm_list = [catalog[0]['name'], catalog[1]['name']]
        else:
            df_list = [main_only, comp_only]
            nm_list = [catalog[0]['name'], catalog[1]['name']]
    else:
        if use_common:
            if use_main:
                df_list = [main_common]
                nm_list = [catalog[0]['name']]
            else:
                df_list = [comp_common]
                nm_list = [catalog[1]['name']]
        else:
            if use_main:
                df_list = [main_only]
                nm_list = [catalog[0]['name']]
            else:
                df_list = [comp_only]
                nm_list = [catalog[1]['name']]
    return df_list, nm_list

def plot_scenario(catalog: dict[int, dict[str, Any]], tol: int, catalog_range: dict[str, float],
                   axes: Axes, geo_ax: GeoAxes, use_both: bool, use_main: bool, use_common: bool):
    
    df_list, nm_list = plot_bypass(catalog, tol, use_both, use_main, use_common)
    for i, (catalog_df, equip) in enumerate(zip(df_list, nm_list)):

        catalog_df = catalog_filter(
            catalog_df=catalog_df, catalog_range=catalog_range
        )

        geo_ax.scatter(
            catalog_df["longitude"],
            catalog_df["latitude"],
            s=5,
            c="b" if i == 1 else "r",
            alpha=0.5,
            label=f"{equip}\ncommon event num: {len(catalog_df)}",
            rasterized=True
        )

        axes[0, 1].scatter(
            catalog_df["depth_km"],
            catalog_df["latitude"],
            s=5,
            c="b" if i == 1 else "r",
            alpha=0.5,
            label=equip,
            rasterized=True,
        )

        axes[1, 0].scatter(
            catalog_df["longitude"],
            catalog_df["depth_km"],
            s=5,
            c="b" if i == 1 else "r",
            alpha=0.5,
            label=equip,
            rasterized=True,
        )

def plot_station(all_station_info: Path, geo_ax):
    """
    plot the station distribution
    """
    station_info = pd.read_csv(all_station_info)
    mask_digit = station_mask(station_info)
    mask_non_digit = ~mask_digit
    if mask_digit.any():
        geo_ax.scatter(
            station_info[mask_digit]["longitude"], 
            station_info[mask_digit]["latitude"], 
            s=2, 
            c="k", 
            marker=".", 
            alpha=0.5, 
            rasterized=True,
            label='DAS'
            )
    if mask_non_digit.any():
        geo_ax.scatter(
            station_info[mask_non_digit]["longitude"], 
            station_info[mask_non_digit]["latitude"], 
            s=100, 
            c="c", 
            marker="^", 
            alpha=0.7, 
            rasterized=True,
            label='Seismometer'
            )
        
def plot_distribution(catalog: dict[int, dict[str, Any]], all_station_info: Path,
                      map_range: dict[str, float], catalog_range: dict[str, float], tol: int,
                      use_ori: bool, use_both: bool, use_common: bool, use_main: bool, figure_name: str
                      ) -> None:
    """
    main ploting function
    """
    params = {
        "font.size": 18,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "legend.fontsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "lines.linewidth": 3,
        "lines.markersize": 10,
        'image.origin': "lower",
        "figure.figsize": (4 * 2.5, 3 * 2.5),
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    }
    matplotlib.rcParams.update(params)
    ls = LightSource(azdeg=0, altdeg=45)
    fig, axes = plt.subplots(
        2, 2, figsize=(12, 12 * (map_range["max_lat"]-map_range["min_lat"])
                       /((map_range["max_lon"]-map_range["min_lon"])
                         *np.cos(np.deg2rad(map_range["min_lat"])))), 
                         gridspec_kw={'width_ratios': [3, 1], 'height_ratios': [3, 1]}
                         )
    geo_ax = fig.add_subplot(2, 2, 1, projection=ccrs.PlateCarree())
    region = [map_range["min_lon"], map_range["max_lon"], map_range["min_lat"], map_range["max_lat"]]
    topo = pygmt.datasets.load_earth_relief(resolution="15s", region=region).to_numpy() / 1e3  # km
    x = np.linspace(map_range["min_lon"], map_range["max_lon"], topo.shape[1])
    y = np.linspace(map_range["min_lat"], map_range["max_lat"], topo.shape[0])
    dx, dy = 1, 1
    xgrid, ygrid = np.meshgrid(x, y)

    geo_ax.pcolormesh(
        xgrid,
        ygrid,
        ls.hillshade(topo, vert_exag=10, dx=dx, dy=dy),
        vmin=-1,
        shading="gouraud",
        cmap="gray",
        alpha=1.0,
        antialiased=True,
        rasterized=True
    )
    geo_ax.coastlines(resolution='10m', color='black', linewidth=1)
    geo_ax.add_feature(cfeature.BORDERS, linestyle=':')
    # geo_ax.add_feature(cfeature.LAND)
    # geo_ax.add_feature(cfeature.OCEAN)

    # You can continue to plot other subplots and customize them as needed
    # Example of setting extent and gridlines
    geo_ax.set_extent(region, crs=ccrs.PlateCarree())
    gl = geo_ax.gridlines(draw_labels=True)
    gl.top_labels = False  # Turn off top labels
    gl.right_labels = False

    if use_ori:
        plot_ori(
            catalog=catalog, 
            catalog_range=catalog_range,
            use_both=use_both,
            use_main=use_main,
            axes=axes,
            geo_ax=geo_ax
            )
    else:
        plot_scenario(
            catalog=catalog, 
            tol=tol,
            catalog_range=catalog_range,  
            axes=axes,
            geo_ax=geo_ax,
            use_both=use_both,
            use_common=use_common,
            use_main=use_main
            )
        
    plot_station(all_station_info, geo_ax)
    # geo_ax.set_title
    geo_ax.autoscale(tight=True)
    xlim = geo_ax.get_xlim()
    ylim = geo_ax.get_ylim()
    # geo_ax.set_aspect(1.0/np.cos(np.deg2rad(min_lat)))
    geo_ax.set_aspect("auto")
    # geo_ax.set_xlim(xlim)
    # geo_ax.set_ylim(ylim)
    geo_ax.legend(markerscale=2) # markerscale=5
    # geo_ax.set_ylabel("Latitude") 
    axes[0,1].autoscale(tight=True)
    axes[0,1].set_ylim(ylim)
    axes[0,1].set_xlim([0, map_range["max_depth"]+1])
    axes[0,1].set_xlabel("Depth (km)")
    axes[0,1].set_ylabel("Latitude")

    axes[1,0].autoscale(tight=True)    
    axes[1,0].set_xlim(xlim)
    axes[1,0].set_ylim([0, map_range["max_depth"]+1])
    axes[1,0].invert_yaxis()
    axes[1,0].set_ylabel("Depth (km)")
    axes[1,0].set_xlabel("Longitude")
    axes[1, 1].axis('off')
    axes[0, 0].axis('off')
    plt.tight_layout()
    # *** haven't finish the customized figure_name
    plt.savefig(figure_path / figure_name) # using fig.savefig when we have several figs.

if __name__ == '__main__':
    # output setting
    figure_path = Path('/home/patrick/Work/EQNet/tests/hualien_0403/compare')
    figure_path.mkdir(parents=True, exist_ok=True)
    # station
    seis_station_info = Path("/home/patrick/Work/EQNet/tests/hualien_0403/station_seis.csv")
    all_station_info = Path("/home/patrick/Work/EQNet/tests/hualien_0403/station_all.csv")
    das_station_info = Path("/home/patrick/Work/EQNet/tests/hualien_0403/station_das.csv")
    # catalog
    seis_catalog = Path("/home/patrick/Work/AutoQuake/GaMMA/results/Hualien_data/daily/20240403.csv")
    das_catalog = Path("/home/patrick/Work/EQNet/tests/hualien_0403/gamma_das/gamma_order.csv")
    combined_catalog = Path("/home/patrick/Work/EQNet/tests/hualien_0403/gamma_seis_das/gamma_order.csv")
    das_threshold_catalog = Path("/home/patrick/Work/EQNet/tests/hualien_0403/check_fig/das_threshold.csv")
    new_das = Path("/home/patrick/Work/EQNet/tests/hualien_0403/gamma_test/test_4/events_66_22.csv")
    # h3dd catalog
    combined_h3dd = Path("/home/patrick/Work/AutoQuake/Reloc2/seis_das_whole.dat_ch.hout")
    seis_h3dd = Path("/home/patrick/Work/EQNet/tests/hualien_0403/20240403.hout")
    # 2024/04/01-2024/04/17 mag catalog from CWA, comparing it to GaMMA
    figure_path = Path('/home/patrick/Work/playground/cwa_gamma/5s/fig')
    cwa_all = Path("/home/patrick/Work/playground/cwa_gamma/cwa_all.csv")
    gamma_all = Path("/home/patrick/Work/playground/cwa_gamma/gamma_all.csv")

    # packing the data, you can only put 1 in it.
    catalog_list = [gamma_all, cwa_all] # [the main catalog, another catalog you want to compare]
    name_list = ["GaMMA(0401-0417)", "CWA(0401-0417)"] # ["name of main catalog", "name of compared catalog"]
    catalog_dict = pack(catalog_list, name_list) # pack it as a dictionary

    # map range on axes
    ## basically, map range should larger than catalog_range.
    map_range = {
        'min_lon': 119.8,
        'max_lon': 123.5,
        'min_lat': 21.5,
        'max_lat': 25.4,
        'min_depth': 0,
        'max_depth': 96
        }
    # catalog range for filtering events
    catalog_range = {
        'min_lon': 119.8,
        'max_lon': 123.5,
        'min_lat': 21.5,
        'max_lat': 25.4,
        'min_depth': -0.1,
        'max_depth': 96
    }
    # time residual between main event and event being compared.
    tol = 5

    # Scenarios
    '''
    example scenario: plot only the compared catalog unique events distribution
    use_ori = False (without any preprocessing)
    use_both = False (both/single)
    use_common = False (common/unique)
    use_main = False (main/compared)
    '''
    use_ori =  True
    use_both = True
    use_common =  False
    use_main = False

    # status check and figure name decide.
    figure_name = status_message_and_fig_name(
        catalog_list=catalog_list, name_list=name_list,
        use_ori=use_ori, use_both=use_both,
        use_common=use_common, use_main=use_main
        )
    
    # main program
    plot_distribution(
        catalog=catalog_dict, all_station_info=all_station_info,
        map_range=map_range, catalog_range=catalog_range,
        tol=tol, use_ori=use_ori, use_both=use_both,
        use_common=use_common, use_main=use_main, figure_name=figure_name
        )
# %%