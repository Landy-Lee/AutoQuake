#%%
import os
import pygmt
from collections import defaultdict
from typing import Dict

def focal_reader(focal_file):
    focal = defaultdict()
    with open(focal_file, 'r') as r:
        lines = r.readlines()
        box = []
        for i, line in enumerate(lines):
            part = line.strip().split()
            lon = float(part[2])
            lat = float(part[3])
            depth = float(part[4])
            mag = float(part[5])
            strike = int(part[6].split('+')[0])
            dip = int(part[8].split('+')[0])
            rake = int(part[10].split('+')[0])
            focal[i] = {'focal_mechanism':{"strike": strike, "dip": dip, "rake": rake, "magnitude": mag}, 'lon': lon, 'lat': lat, 'depth': depth}
            box.append(i)
    return focal, max(box)

def plot_meca(focal_dict, event_num):
    # plot the focal on Taiwan
    fig = pygmt.Figure()
    fig.coast(
        region=region,
        projection="M6c",
        land="grey",
        water="lightblue",
        shorelines=True,
        frame="a"
    )
    for i in range(event_num):
        fig.meca(
            spec=focal_dict[i]['focal_mechanism'],
            scale=f"{meca_size}c+m",  # in centimeters
            longitude=focal_dict[i]['lon'],
            latitude=focal_dict[i]['lat'],
            depth=focal_dict[i]['depth'],
            # Fill compressive quadrants with color "red"
            # [Default is "black"]
            compressionfill="red",
            # Fill extensive quadrants with color "cornsilk"
            # [Default is "white"]
            extensionfill="cornsilk",
            # Draw a 0.5 points thick dark gray ("gray30") solid outline via
            # the pen parameter [Default is "0.25p,black,solid"]
            pen="0.5p,gray30,solid",
        )
    fig.text(x=max(region)-0.01, y=min(region)+0.02, text=f"Beachball amount: {event_num}", font="6p,Helvetica-Bold,black", justify="RB")
    #fig.show()
    fig.savefig(os.path.join(parent_dir, f'{source}_focal_taiwan.png'), dpi=300)


if __name__ == '__main__':
    # plotting the focal_mechanism  
    meca_size = '0.2'
    region=[119.7, 122.5 , 21.7 , 25.4] # whole Taiwan
    #region=[121, 122.5 , 23.6 , 24.4] # Hualien area
    source = 'gamma'
    parent_dir = '/home/patrick/Work/playground/fig'
    focal_result = f'/home/patrick/Work/playground/{source}_gafocal_20240401_20240417_results.txt'
    focal_dict, event_num = focal_reader(focal_result)
    plot_meca(focal_dict, event_num)
    # plot the Mag distribution during 20240401-20240417
# %%
