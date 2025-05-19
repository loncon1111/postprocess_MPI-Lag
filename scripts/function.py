# import libraries

import alphashape
import shapely.geometry as geometry
from shapely.geometry import Polygon
from descartes import PolygonPatch
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import netCDF4
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import shape
from shapely.ops import transform
import pyproj
from functools import partial
from datetime import datetime, timedelta

# define some functions
def ow_lag(date):
    # Read from forward
    f_fw = netCDF4.Dataset('../data/forward/new_%s.nc' %date, 'r')
    f_bw = netCDF4.Dataset('../data/backward/new_%s.nc' %date, 'r')

    f_ieig1 = f_fw.variables['ieig1']
    f_reig1 = f_fw.variables['reig1']
    b_ieig1 = f_bw.variables['ieig1']
    b_reig1 = f_bw.variables['reig1']


    ieig1 = f_ieig1[:,:,:] + b_ieig1[:,:,:]
    reig1 = f_reig1[:,:,:] + b_reig1[:,:,:]

    ow_lag = ieig1[:,:,:] - b_ieig1[:,:,:]
    # Flush
    f_fw = None; f_bw = None; f_ieig1 = None; f_reig1 = None; b_ieig1 = None; b_reig1 = None
    ieig1 = None; reig1 = None

    return ow_lag

def vort(date):
    # Read from forward
    f_fw = netCDF4.Dataset('../data/forward/vort_%s.nc' %date, 'r')
    f_bw = netCDF4.Dataset('../data/backward/vort_%s.nc' %date, 'r')

    f_vort = f_fw.variables['relvort']
    b_vort = f_bw.variables['relvort']

    relvort = f_vort[:,:,:] + b_vort[:,:,:]

    # Flush
    f_fw = None; f_bw = None

    return relvort

def get_dims(date):
    # read netCDF file
    f_fw = netCDF4.Dataset('../data/forward/vort_%s.nc' %date, 'r')
    lon  = f_fw.variables["longitude"][:]
    lat  = f_fw.variables["latitude"][:]
    lev  = f_fw.variables["level"][:]
    return lon,lat,lev
    

def next_time(startdate,days=0,hours=0,minutes=0,seconds=0):
    """Find next time for calculation"""
    sdate = datetime.strptime(startdate, '%Y%m%d_%H')
    date = sdate + timedelta(days=days, hours=hours)
    date = datetime.strftime(date, '%Y%m%d_%H')
    return date

def duration(startdate,enddate):
    """Time duration in hours"""
    sdate = datetime.strptime(startdate, '%Y%m%d_%H')
    edate = datetime.strptime(enddate  , '%Y%m%d_%H')
    delta = edate - sdate
#     if isinstance(delta, np.timedelta64):
#         return delta.astype(timedelta).total_hours() / 60.
    return delta.days, delta.seconds


