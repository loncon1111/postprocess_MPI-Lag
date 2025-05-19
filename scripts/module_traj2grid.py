import numpy as np
import netCDF4
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import integrate
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import class_plot_the as func_mask
from new_func_lcs import FTLE
import numpy_indexed as npi

def get_dataset(ncname):
    raw_ds = func_mask._traj_array(ncname = ncname)
    xx0 = np.unique(raw_ds['lon'][:,0])
    yy0 = np.unique(raw_ds['lat'][:,0])
    zz0 = np.unique(raw_ds['p'][:,0])
    x0, y0, z0 = np.meshgrid(xx0,yy0,zz0)


    shape = np.shape(x0)
    ntra,ntime = func_mask._get_netcdf_traj_dim(ncname)
    l = list(shape); l.append(ntime)

    dim_arr = np.stack((x0*1e8,y0*1e4,z0/100), axis = -1)
    print(dim_arr.shape)
    
    print("check")
    print(sum(dim_arr[0,0,-1,:]))
    
    # do the trick here: sum over the last dim
    dim_arr = dim_arr.sum(axis = -1, dtype = np.float64)
    print(dim_arr[0,0,-1])
    print(dim_arr.shape)

    flat_arr = np.ravel(dim_arr, order = 'C') # do the flatten so as you can get list of indices

    traj_dim = np.dstack(
                        (raw_ds["lon"][:,0]*1e8, 
                         raw_ds["lat"][:,0]*1e4,
                         raw_ds["p"][:,0]*1e-2
                        )
    ) # it is the first position

    traj_dim = np.squeeze(traj_dim)
    traj_dim = traj_dim.sum(axis = -1, dtype = np.float64) # also the trick: sum over the last dim --> reduce last dim
    print(np.unique(traj_dim).shape)

    new_ds = np.empty(l, dtype = raw_ds.dtype)

    flat_idx = npi.indices(flat_arr,traj_dim) # this package is stupid, but save the time (only work for simple cases)
    ix,iy,iz = np.unravel_index(flat_idx,dim_arr.shape, order = 'C') # convert index of flatten to unflatten array
    print(ix.shape,iy,iz)
    print(raw_ds.shape)
    new_ds[ix,iy,iz,:] = raw_ds # simple, smart, and quick

    for iname,name in enumerate(new_ds.dtype.names):
        #print(new_ds[name][ix[100],iy[100],iz[100],:] == raw_ds[name][100,:])
        dat = xr.DataArray(
            data = new_ds[name],
            dims = ["yy0","xx0","zz0","time"],
            coords = {
                'xx0' : xx0,
                'yy0' : yy0,
                'zz0' : zz0,
                'time' : np.arange(0,ntime*6,6)
            },
            name = name
        )
        if iname == 0:
            ds = dat.to_dataset()
        else:
            ds = xr.merge([ds,dat])

    ds = ds.rename(name_dict = {
        'lon' : 'x',
        'lat' : 'y',
        'p'   : 'z'
        } )
    ds["x0"] = xr.DataArray(data = x0, dims = ["yy0","xx0","zz0"])
    ds["y0"] = xr.DataArray(data = y0, dims = ["yy0","xx0","zz0"])
    ds["z0"] = xr.DataArray(data = z0, dims = ["yy0","xx0","zz0"])

    return ds