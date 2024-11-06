#%%
import os
import pygmt 
import pandas as pd 

def plot_mag():
    df = pd.read_csv(catalog_data, sep='\s+', header=None)
    if source == 'cwb':
        df.columns = ['date', 'time', 'latitude', 'longitude', 'depth', 'magnitude', 'other_1','other_2','other_3','other_4']
    else:
        df.columns = ['date', 'time', 'latitude', 'longitude', 'depth', 'magnitude']
    #df =df[df['magnitude'] > 4]
    fig = pygmt.Figure()
    fig.basemap(region=region, projection="M15c", frame=True)
    fig.coast(land="white", water="skyblue")
    pygmt.makecpt(cmap="viridis", series=[df['depth'].min(), df['depth'].max()])
    fig.plot(
        x=df['longitude'],
        y=df['latitude'],
        size=0.02 * 2**df['magnitude'],
        fill=df['depth'],
        cmap=True,
        style="cc",
        pen="black",
    )
    fig.colorbar(frame="xaf+lDepth (km)")
    fig.savefig(os.path.join(parent_dir, f'{source}_mag_pygmt.png'), dpi=300)
    
if __name__ == '__main__':
    parent_dir = '/home/patrick/Work/playground/fig'
    catalog_data = '/home/patrick/Work/playground/cwb_catalog_20240401_20240417.gmt'
    source='cwb'
    region=[119.7, 122.5 , 21.7 , 25.4]
    plot_mag()
# %%
