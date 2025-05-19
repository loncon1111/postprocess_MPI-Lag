# coding=utf-8

#%%
import netCDF4
import numpy as np
import pandas as pd
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridsprec
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredText
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import geopandas as gpd

# %%
def next_time(startdate,days=0,hours=0,minutes=0,seconds=0):
    """
    Find next time for calculation
    """
    sdate = datetime.strptime(startdate, '%Y%m%d_%H')
    date  = sdate + timedelta(days=days, hours=hours)
    date  = datetime.strftime(date, "%Y%m%d_%H")
    return date

def duration(startdate,enddate):
    """
    Time duration in hours
    """
    sdate = datetime.strptime(startdate, "%Y%m%d_%H")
    edate = datetime.strptime(enddate,   "%Y%m%d_%H")
    delta = edate - sdate
    return delta.days, delta.seconds
#%%
cdir = "/Users/daohoa/Desktop/my-notebook/LAGRANGIAN/TSCOMBO2017/results/"
data_dir = "/Users/daohoa/Desktop/my-notebook/LAGRANGIAN/TSCOMBO2017/data/"

# %%
sdate = "20170716_06"
edate = "20170716_06"

dur_day, dur_sec = duration(sdate,edate)
dur_hr = dur_sec // (60*60) + dur_day * 24
ndate = dur_hr // 6 # time interval

# %%
rv_date = sdate
level   = 500
# %%
df = pd.read_csv(cdir + "label_list2d_%s_%s_vort.csv" %(rv_date,level),index_col=0)

index = 0

# %% 
df

# %%
# Open cc2d file to get all the coordinates of the vortex
ncdir = "/Users/daohoa/Desktop/my-notebook/LAGRANGIAN/TSCOMBO2017/CLU_VORT/nc_files/"
ncfile = netCDF4.Dataset(ncdir + "cc2d_%s.nc" %rv_date)

lons = ncfile.variables["longitude"][:]
lats = ncfile.variables["latitude"][:]
levs = ncfile.variables["level"][:]

ref_label = df["label"].iloc[index]
ref_value = df["value"].iloc[index]

ilev = np.squeeze(np.where(levs == level))
print(ilev)

var = ncfile.variables["labels_cc2d_%05.2f" %ref_value][ilev,:,:]


ix, iy = np.where(var == ref_label)

ref_lon = np.take(lons,ix)
ref_lat = np.take(lats,iy)
ref_lev = level

ntrajs = ref_lon.size

# %%
### Open startf file ###
srcdir = "/Users/daohoa/Desktop/my-notebook/LAGRANGIAN/TSCOMBO2017/data/backward/"
src_file = netCDF4.Dataset(srcdir + "startf_%s.4" %rv_date)
#print(src_file.variables.items())

src_lon = np.squeeze(src_file.variables["lon"][:])
src_lat = np.squeeze(src_file.variables["lat"][:])
src_lev = np.squeeze(src_file.variables["level"][:])

arr = np.empty(0,dtype = int)
for item in range(ntrajs):
    arr = np.append(arr, np.where((src_lon == ref_lon[item]) & 
                       (src_lat == ref_lat[item]) &
                       (src_lev == ref_lev)))
    


# %%

# Writing new netCDF4 file
oufile = netCDF4.Dataset(cdir + "startf_%s.4" %rv_date, mode='w')
   
# copy global attributes all at once via dictionary
oufile.setncatts(src_file.__dict__)

# copy dimensions except for dimx_lon
oufile.createDimension('dimx_lon',ntrajs)
for name, dimension in src_file.dimensions.items():
    if name != 'dimx_lon':
        print(name, dimension)
        oufile.createDimension(name, (len(dimension)) if not dimension.isunlimited() else None)
    
# copy all file data except for the excluded

# time
for name, variable in src_file.variables.items():
    print(name, variable, variable.dimensions)
    oufile.createVariable(name, variable.datatype, variable.dimensions)
    
    if (name != 'time') & (name != 'BASEDATE'):
        for iarr,rarr in enumerate(arr):
            oufile[name][:,:,:,iarr] = src_file[name][:,:,:,rarr]
    else:
        oufile[name][:] = src_file[name][:]
        

    # copy variable attributes all at once via dictionary
    oufile[name].setncatts(src_file[name].__dict__)

# %%
print(oufile.variables['BASEDATE'][:])
# %%
oufile.close()
src_file.close()
# %%
print(rv_date)
# %%
