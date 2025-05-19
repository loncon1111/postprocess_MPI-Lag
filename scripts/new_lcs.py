#coding=UTF-8

# %%
# import libraries
from re import S
import numpy as np
import netCDF4
import pandas as pd
import datetime
from scipy import integrate
import xarray as xr
import class_plot_the as func_mask
from new_func_lcs import FTLE
import numpy_indexed as npi
from module_traj2grid import get_dataset
import sys, getopt

def run_ftle_with_file(trajfile):
    ds = get_dataset(trajfile)
    ds = ds.transpose("zz0","time", "yy0", "xx0")
    FTLE_extractor = FTLE(
        spherical_flag = True,
        flag_3d = False,
        integration_time_index = -1,
        dim_x = -1,
        dim_y = -2,
        dim_z = None
    )
    ds_output = FTLE_extractor.explore_ftle_2d_vertical(ds, to_dataset = True)
    return ds_output

# %%
date = ''
try:
   opts, args = getopt.getopt(sys.argv[1:],"hi:", "input_date=")
except getopt.GetoptError:
   print('python new_lcs.py -i <input_date>')
   sys.exit(2)

for opt, arg in opts:
   if opt == '-h':
       print('python new_lcs.py -i <input_date>')
       sys.exit()
   elif opt in ("-i", "--input_date"):
       date = arg

print('input_date is :', date)

ncdir = "/work/users/hoadnq/lazy_make_it_better/data/"

fwd_ncdir = ncdir + "forward/"
bwd_ncdir = ncdir + "backward/"

#### Forward
# Read netCDF file
fwd_ncname = fwd_ncdir + "lsl_%s.4" %date 
fwd_ds = run_ftle_with_file(fwd_ncname) 
fwd_ds.to_netcdf(fwd_ncdir + 'ftle_fwd_%s.nc' %date)

#### Backward
bwd_ncname = bwd_ncdir + "lsl_%s.4" %date
bwd_ds = run_ftle_with_file(bwd_ncname)
bwd_ds.to_netcdf(bwd_ncdir + 'ftle_bwd_%s.nc' %date)

exit()
