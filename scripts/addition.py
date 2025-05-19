#!/usr/bin/env python

#coding:utf-8

# import libs
import numpy as np
import netCDF4
import xarray as xr
import pandas as pd
import glob
from datetime import timedelta,datetime
import class_plot_the as func_traj
import sys, os

work_dir = "/work/users/hoadnq/lazy_make_it_better/"
cyc_dir = work_dir + "vortex_regions/"
excel_dir = work_dir + "similarity/"
dat_dir = work_dir + "data/add/"

#cyclone = "TALAS"
#typical_trajs = [931,460]
#date    = "20170715_00"

# read trajectories
#lsl_file = glob.glob(cyc_dir+"lsl_%s_%s_clt.4" %(date,cyclone))[0]

all_lsl = glob.glob(cyc_dir + "lsl*_clt.4")
print(all_lsl)
for lsl_file in all_lsl:
    baselsl = os.path.basename(lsl_file).split('_')
    #cyclone = baselsl[3]
    date = "%s_%s" %(baselsl[1],baselsl[2])


    # get traj array
    traj_array = func_traj._traj_array(lsl_file)

    lons = traj_array["lon"]
    lats = traj_array["lat"]

    ntra,ntime = lons.shape
    times = np.linspace(0,-72,num=ntime)
    print(times)

    sdate = datetime.strptime(date,"%Y%m%d_%H")

    the = np.empty(traj_array.shape, dtype = np.float64)
    divg = np.empty(traj_array.shape, dtype = np.float64)


    for itime, time in enumerate(times):
        date = sdate + timedelta(hours = time)
        ds = xr.open_dataset(dat_dir + "P%s" %date.strftime("%Y%m%d_%H"))
        lon = lons[:,itime]; lat = lats[:,itime]
        the[:,itime] = ds["the"].interp(
            longitude = lon,
            latitude = lat,
            method = "linear").values.diagonal().squeeze()
        divg[:,itime] = ds["divg"].interp(
            longitude = lon,
            latitude = lat,
            method = "linear").values.diagonal().squeeze()
        ds.close()

    #get very raw data

    ncfile = netCDF4.Dataset(lsl_file,mode="r+")
    ref_var = ncfile.variables["lon"]

    try:
        equiv_th = ncfile.createVariable('the',np.float64,ref_var.dimensions)
        equiv_th[:] = the.reshape(ref_var.shape)
        #copy variable attributes
        equiv_th.setncatts(ref_var.__dict__)
    except RuntimeError:
        pass

    try:
        diver = ncfile.createVariable('divg',np.float64,ref_var.dimensions)
        diver[:] = divg.reshape(ref_var.shape)
        diver.setncatts(ref_var.__dict__)
    except RuntimeError:
        pass

    print(ncfile)

    ncfile.close()
    # Flush
    lons,lats,equiv_th,the,divg,diver = 6*[None]
