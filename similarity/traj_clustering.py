#/usr/env python

#coding:utf-8

# import libs
import numpy as np
import xarray as xr
import netCDF4
import glob
import pandas as pd
import class_plot_the as func_traj
import similaritymeasures
import function_ml

# 
workdir = "/work/users/hoadnq/lazy_make_it_better/"
traj_dir = workdir + "vortex_regions/"

cyc_name = "TALAS"
date = "20170715_00"

lsl_file = glob.glob(traj_dir + "lsl_%s_%s_clt.4" %(date,cyc_name))[0]

# read trajectories
traj_array = func_traj._traj_array(lsl_file)

level = 850

traj = traj_array[traj_array['p'][:,0] == level]
print(traj.shape)
ntraj, ntime = traj.shape
print(len(traj))

# get the distance matrix
traj_locs = function_ml._get_trajectories(traj)
print(traj_locs)
dist_m = function_ml._distance_matrix(traj_locs, method = "Frechet")

print(dist_m)



