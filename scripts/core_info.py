#!/usr/bin/env python

#%% import libs
import numpy as np
import pandas as pd
import xarray as xr
import glob
import alphashape
import pyproj
from shapely.geometry import shape
from shapely.ops import transform

import class_plot_the as traj_func
#import geopandas as gpd
from functools import partial
# %%
if __name__ == "__main__":
    # input
    cyc_name = "HAITANG"
    # directories
    #wdir = "/Users/daohoa/Desktop/my-notebook/LAGRANGIAN/TSCOMBO2017/"
    wdir = "/work/users/hoadnq/lazy_make_it_better/"
    cdir = wdir + "results/"
    cyc_dir = wdir + "vortex_regions/"
    data_dir = wdir + "data/"
    plot_dir = wdir + 'plot/'
    inf_dir = wdir + 'label_first/'
    # Define projection to calculate area
    proj = partial(pyproj.transform, pyproj.Proj(init='epsg:4326'),
               pyproj.Proj(init='epsg:3857'))
    # %%
    all_avail =  sorted(glob.glob(inf_dir + '%s/*.xlsx' %cyc_name))
    for ifile, s in enumerate(all_avail):
        df = pd.read_excel(s)
        #df['area'] = np.nan; df['perimeter'] = np.nan
        #df['bnds_lon'] = np.nan; df['bnds_lat'] = np.nan
        level = df['level'].to_numpy().squeeze()
        print(level)
        date = s[s.find('label_first_')+len('label_first_'):s.rfind('_'+cyc_name)]
        print(date)
        era5 = xr.open_dataset(data_dir + 'ERA5/P%s' %date)
        q = era5['Q'].sel(level=level)
        mask = traj_func._init_mask(q); mask = mask.squeeze()
        track_file = glob.glob(cyc_dir + 'ofile_%s_%s_clt.4' %(date,cyc_name))[0]
        mask = traj_func._get_mask(ncname = track_file,
                                   mask_array = mask)

        hull_area = np.empty(0); hull_perim = np.empty(0)
        for ind,row in df.iterrows():
            lev = row['level']
            submask = mask.sel(level = lev).astype('bool')
            lonind,latind = np.where(submask)
            lons = submask.longitude[lonind].values
            lats = submask.latitude[latind].values
            points = np.stack([lons,lats]).T
        
            # get convexhull
            if lons.size != 0:
                #alpha = 0.2
                alpha = alphashape.optimizealpha(points)          
                hull = alphashape.alphashape(points, alpha)
                hull_lons, hull_lats = hull.exterior.coords.xy
                s = shape(hull)
                hull_area = np.append(hull_area,transform(proj, s).area)
                hull_perim = np.append(hull_perim,transform(proj, s).length)
                #p = gpd.GeoSeries(hull)
                #p.plot()
            else:
                hull_area = np.append(hull_area, np.nan)
                hull_perim = np.append(hull_perim, np.nan)

            
        df['hull_area'] = hull_area; df['hull_perim'] = hull_perim
        df['symmetry'] = 1 + np.arctan(np.pi*(4*np.pi*hull_area/np.power(hull_perim,2) - 1))
        df.to_excel('shape_%s_%s.xlsx' %(date,cyc_name))
            
        df = None; hull_area = None; hull_perim = None; q = None

