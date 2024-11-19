#%%
from pathlib import Path
import time
import pandas as pd
from datetime import datetime
import pygmt 

def compare(df_main, df_compare):
    rows_with_sign = []
    for _, main_row in df_main.iterrows():
        match_found = False
        for _, compare_row in df_compare.iterrows():
            if abs((main_row['datetime'] - compare_row['datetime']).total_seconds()) < 5: # when we find this, actually we can directly plot the meca
                row_with_sign = main_row.to_dict()
                row_with_sign['match'] = 1
                rows_with_sign.append(row_with_sign)              
                match_found = True
                break
        if not match_found:
            # Append the non-matching row from df_cwa with sign 0
            row_with_sign = main_row.to_dict()
            row_with_sign['match'] = 0
            rows_with_sign.append(row_with_sign)
    
    # Convert the list to a new DataFrame
    df_new = pd.DataFrame(rows_with_sign)
    return df_new

def find_event():
    df_cwa = pd.read_csv(cwa_path, sep='\s+', header=None)
    df_cwa['datetime'] = pd.to_datetime(df_cwa[0] + ' ' + df_cwa[1])
    df_gamma = pd.read_csv(gamma_path, sep='\s+', header=None)
    df_gamma['datetime'] = pd.to_datetime(df_gamma[0] + ' ' + df_gamma[1])
    
    # apply funcion
    df_cwa_new = compare(df_cwa, df_gamma)
    df_gamma_new = compare(df_gamma, df_cwa)
    # Output the DataFrame to a CSV file
    #df_cwa_new.to_csv(parent_dir / 'cwa_output.csv', index=False)
    #df_gamma_new.to_csv(parent_dir / 'gamma_output.csv', index=False)
    return df_cwa_new, df_gamma_new

# plotting
def dict_create(row):
    dict_name = {
            'focal_mechanism':{"strike": int(row[6].split('+')[0]),
                                "dip": int(row[8].split('+')[0]), 
                                "rake": int(row[10].split('+')[0]), 
                                "magnitude": float(row[5])},
            'lon': float(row[2]), 
            'lat': float(row[3]), 
            'depth': float(row[4])
                    }
    return dict_name

def plot_meca(cwa_dict, gamma_dict, count):
    fig = pygmt.Figure()
    # cwa
    fig.coast(
        region=region,
        projection="M6c",
        land="grey",
        water="lightblue",
        shorelines=True,
    )
    fig.basemap(frame=["WSne+tCWA", "xa2f0.5", "ya21f0.5"])
    fig.meca(
        spec=cwa_dict['focal_mechanism'],
        scale=f"{meca_size}c+m",  # in centimeters
        longitude=cwa_dict['lon'],
        latitude=cwa_dict['lat'],
        depth=cwa_dict['depth'],
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
    fig.text(x=max(region)-0.01, y=min(region)+0.02, text=f"Hi", font="6p,Helvetica-Bold,black", justify="RB")
    fig.shift_origin(xshift="w+0.5c")
    # gamma
    fig.coast(
        region=region,
        projection="M6c",
        land="grey",
        water="lightblue",
        shorelines=True,
    )
    fig.basemap(frame=["wSne+tGaMMA", "xa2f0.5", "ya1f0.5"])
    fig.meca(
        spec=gamma_dict['focal_mechanism'],
        scale=f"{meca_size}c+m",  # in centimeters
        longitude=gamma_dict['lon'],
        latitude=gamma_dict['lat'],
        depth=gamma_dict['depth'],
        # Fill compressive quadrants with color "red"
        # [Default is "black"]
        compressionfill="blue",
        # Fill extensive quadrants with color "cornsilk"
        # [Default is "white"]
        extensionfill="cornsilk",
        # Draw a 0.5 points thick dark gray ("gray30") solid outline via
        # the pen parameter [Default is "0.25p,black,solid"]
        pen="0.5p,gray30,solid",
    )
    fig.text(x=max(region)-0.01, y=min(region)+0.02, text=f"Hi", font="6p,Helvetica-Bold,black", justify="RB")
    #fig.show()
    fig.savefig(parent_dir / f'focal_compare_{count}.png', dpi=300)

def plot():
    df_cwa = pd.read_csv(cwa_path, sep='\s+', header=None)
    df_cwa['datetime'] = pd.to_datetime(df_cwa[0] + ' ' + df_cwa[1])
    df_gamma = pd.read_csv(gamma_path, sep='\s+', header=None)
    df_gamma['datetime'] = pd.to_datetime(df_gamma[0] + ' ' + df_gamma[1])
    count = 0
    for _, cwa_row in df_cwa.iterrows():
        for _, gamma_row in df_gamma.iterrows():
            if abs((cwa_row['datetime'] - gamma_row['datetime']).total_seconds()) < 5: # when we find this, actually we can directly plot the meca            
                count += 1
                cwa_dict = dict_create(cwa_row)
                print(cwa_dict)
                print(type(cwa_dict['focal_mechanism']))
                print(type(cwa_dict['lon']))
                print(type(cwa_dict['lat']))
                print(type(cwa_dict['depth']))
                gamma_dict = dict_create(gamma_row)
                plot_meca(cwa_dict, gamma_dict, count)
                break

if __name__ == '__main__':
    starttime = time.time()
    parent_dir = Path('/home/patrick/Work/playground/focal_compare')
    parent_dir.mkdir(parents=True, exist_ok=True)
    gamma_path = '/home/patrick/Work/playground/gamma_gafocal_20240401_20240417_results.txt'
    cwa_path = '/home/patrick/Work/playground/cwb_gafocal_20240401_20240417_results.txt'
    # plotting the focal_mechanism  
    meca_size = '1.5'
    region=[119.7, 122.5 , 21.7 , 25.4] # whole Taiwan
    #region=[121, 122.5 , 23.6 , 24.4] # Hualien area
    plot()
    endtime = time.time()
    duration = endtime - starttime
    print(f"duration: {duration}")
# %%
