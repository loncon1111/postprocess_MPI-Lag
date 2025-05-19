#coding=utf-8

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
#%%
cdir = "/Users/daohoa/Desktop/my-notebook/LAGRANGIAN/TSCOMBO2017/results/"
data_dir = "/Users/daohoa/Desktop/my-notebook/LAGRANGIAN/TSCOMBO2017/data/"
ncdir = "/Users/daohoa/Desktop/my-notebook/LAGRANGIAN/TSCOMBO2017/CLU_VORT/nc_files/"
# %%
sdate = "20170715_00"
edate = "20170722_00"

# %%
rv_date = sdate
level   = 850
# %%
df = pd.read_csv(cdir + "label_list2d_%s_%s.csv" %(rv_date,level),index_col=0)
src = netCDF4.Dataset(ncdir + "cc2d_%s.nc" %rv_date)

# %%
# Handling Excel file
sorted_flag = df.groupby('hull_flag').count().sort_values(by = ["hull_centx"],
                                ascending = False).index# %%
filter = np.empty(0,dtype=int)
for isort,sort in enumerate(sorted_flag):
    if len(df[df["hull_flag"] == sort]) > 20:
        filter = np.append(filter,sort)

# %%
rlev  = src.variables["level"][:]
rlon  = src.variables["longitude"][:]
rlat  = src.variables["latitude"][:]
ilev  = np.squeeze(np.where(rlev == level))

# %%
df_frst = pd.DataFrame(columns = df.columns)
for iflag, flag in enumerate(filter):
    print(flag)
    df_subflag = df[df["hull_flag"] == flag]
    first_check = df_subflag[df_subflag["owlag_flag"] == 1]
    first_flt = first_check.iloc[0]
    first_unflt = df_subflag.iloc[0]

    # 'Filtered' first
    lonind = np.empty(0,dtype=int)
    latind = np.empty(0,dtype=int)
    value = first_flt["value"]
    label = first_flt["label"]
    target = "flt"

    print(value, label)

    var = src.variables["labels_cc2d_%05.2f" %value][ilev,:,:]

    ndims = np.shape(var)
    lonind,latind = np.where(var == label)

    mask = np.zeros(ndims,dtype=int)
    mask[lonind,latind] = 1

    plt.contour(rlon,rlat,mask)
    plt.show()

    ######## START REPLACING STARTF FILE ###########
    # remove if old file exists
    if os.path.exists(cdir + "startf_%s.4" %rv_date):
        os.remove(cdir + "startf_%s.4" %rv_date)

    src_nc = netCDF4.Dataset(data_dir + "backward/startf_%s.4" %rv_date, mode = "r")
    dst_nc = netCDF4.Dataset(cdir     + "startf_%s_%02d_%s.4" %(rv_date,iflag,target), mode = "w", format = "NETCDF4")

    # length of ntra
    lats,lons = np.where(mask == 1)
    lons = rlon[lons]
    lats = rlat[lats]

    # get source lon, lat ,p, level
    lon_src = np.squeeze(src_nc.variables["lon"][:])
    lat_src = np.squeeze(src_nc.variables["lat"][:])
    lev_src = np.squeeze(src_nc.variables["level"][:])

    arr = np.array([],dtype = int)
    for ipt,rpts in enumerate(lats):
        new = np.where((lon_src == lons[ipt]) \
                    &(lat_src == lats[ipt])\
                    &(lev_src == level))
        arr = np.append(arr, new)
        # Flush
        new = None

    # Copy global attributes all at once via dict
    dst_nc.setncatts(src_nc.__dict__)

    # copy dimensions except for dimx_lon
    dst_nc.createDimension('dimx_lon',arr.shape[0])
    for name, dimension in src_nc.dimensions.items():
        if name != 'dimx_lon':
            dst_nc.createDimension(name, len(dimension) if not dimension.isunlimited() else None)

    for name, variable in src_nc.variables.items():
        dst_nc.createVariable(name, variable.datatype, variable.dimensions)
    
        if (name != 'time') & (name != 'BASEDATE'):
            for iarr,rarr in enumerate(arr):
                dst_nc[name][:,:,:,iarr] = src_nc[name][:,:,:,rarr]
        else:
            dst_nc[name][:] = src_nc[name][:]
        

        # copy variable attributes all at once via dictionary
        dst_nc[name].setncatts(src_nc[name].__dict__)
    # close netCDF files
    src_nc.close(); dst_nc.close()

    # Flush
    df_subflag = None; first_check , first = 2*[None]
    lon, lat = 2*[None]
# %%
print(rv_date)
print(src['BASEDATE'][:])
# %%
