#coding=utf-8
import netCDF4
import numpy as np
import xarray as xr

def _get_netcdf_traj_dim(ncfile):
    """ return number of trajectories (ntraj) and number of time step (ntime)"""

    dim_set = {'dimx_lon', 'id', 'ntra'}
    dim_nc  = set(ncfile.dimensions.keys())

    try:
        ntra_dim = dim_set.intersection(dim_nc).pop()
        ntra = len(ncfile.dimensions[ntra_dim])
    except KeyError:
        raise Exception('Cannot read the number of trajectories, ' +
                        'not one of (' + ' '.join(dim_set) + ')')
            
    try: 
        ntime = len(ncfile.dimensions['time'])
    except KeyError:
        ntime = len(ncfile.dimensions['ntim'])

    return ntra, ntime

