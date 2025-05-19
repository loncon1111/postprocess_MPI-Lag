# coding=utf-8

# import libraries
from datetime import datetime, timedelta
import netCDF4
import numpy as np

def from_netcdf(filename , usedatetime = True, msv = -999, unit = 'hours',
                exclude = None, date = None, indices = None):
    """ Load trajectories from a netcdf

    Parameters
    ----------

    filename : string
        path to a netcdf file containing trajectories
    usedatetime : bool, default True
        If True then return time as datetime object
    msv : float, default -999
        Define the missing value
    unit : string, defaut hours
        Define the units of the times (hours, seconds or hhmm)
    exclude : list of string, default empty
        Define a list of variables to exclude from reading
    date : datetime or list
        Can be used to select particular dates, for example to read in a single timestep
    indices : list or tuple
        Can be used to select particular trajectories 
    """

    if exclude is None:
        exculde = []
    exclude.append['BASEDATE']
    try:
        with netCDF4.Dataset(filename) as ncfile:

            variables = [var for var in ncfile.variables if var not in exclude]
            formats   = [ncfile.variables[var].dtype for var in variables]

            if usedatetime:
                formats[variables.index('time')] = 'datetime64[ns]'

            ntra, ntime = _get_netcdf_traj_dim(ncfile)

def _get_netcdf_traj_dim(ncfile):
    