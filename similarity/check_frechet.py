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
import time

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

exp_data = np.zeros((ntime, 3))
exp_data[:, 0] = traj['lon'][0]
exp_data[:, 1] = traj['lat'][0]
exp_data[:, 2] = traj['p'][0]

num_data = np.zeros((ntime, 3))
num_data[:, 0] = traj['lon'][10]
num_data[:, 1] = traj['lat'][10]
num_data[:, 2] = traj['p'][10]

start_time = time.time()
print("Frechet")
df = similaritymeasures.frechet_dist(exp_data, num_data)

print("--- %s seconds ---" %(time.time() - start_time))



