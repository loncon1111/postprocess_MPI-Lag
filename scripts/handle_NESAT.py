#/usr/env python

#coding:utf-8

#%% import libs
import numpy as np
import xarray as xr
import netCDF4
import pandas as pd
import matplotlib.pyplot as plt
import sys,getopt,glob
import geopandas as gpd
import class_plot_the as func_traj
from shapely.geometry import Point, shape, mapping, Polygon
from shapely.ops import unary_union,cascaded_union
# %%
cyclone = "NESAT"
date    = "20170725_18"
#date    = "20170721_18"
# read distance matrix
#work_dir = "/Users/daohoa/Desktop/my-notebook/LAGRANGIAN/TSCOMBO2017/"
work_dir = "/work/users/hoadnq/lazy_make_it_better/"
dst_file = glob.glob(work_dir + 'similarity/distma/dist_matrix_%s_%s.*' %(cyclone,date))[0]
print(dst_file)
cyc_dir = work_dir + "vortex_regions/"
plt_dir = work_dir + "plot/"
datdir = work_dir + "data/ERA5/"
#cdir = "/Users/daohoa/Desktop/my-notebook/"
shp_dir = work_dir + "shapefiles/"
# %% read traj_array
# read trajectories
traj_file = cyc_dir + "lsl_%s_%s_clt.4" %(date,cyclone)
traj_array = func_traj._traj_array(traj_file)
# %% read shapefile
sf = gpd.read_file(shp_dir + "include_scs/include_scs.shp")

# %%
traj_array['lon'][:,-1]
# %%
ntra, ntime = traj_array.shape
# %%
lon = traj_array['lon'][:,-1]
lat = traj_array['lat'][:,-1]
#point = Point(lon,lat)
#print(len(np.where(sf['geometry'].contains(point))))
points = gpd.points_from_xy(x = lon, y = lat)
points = gpd.GeoSeries(points)
regions = gpd.GeoSeries(sf['geometry'])
# %%
df = pd.DataFrame(columns = ['longitude','latitude'])
df['longitude'] = lon; df['latitude'] = lat

# %%
df['contains'] = points.apply(lambda x: regions.contains(x).any())
#%%
df.to_excel(work_dir + 'NESAT_contains.xlsx',index=False)
