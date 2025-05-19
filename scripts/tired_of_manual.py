#coding=utf-8

#%%
import matplotlib
matplotlib.use('Agg')

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
import function
from matplotlib import cm

#### MPI environments ####
from mpi4py import MPI
#### MPI functions ####
def para_range(n1, n2, nprocs, irank):

    iwork1 = (n2 - n1) // nprocs
    iwork2 = (n2 - n1) % nprocs
    ista   = irank * iwork1 + n1 + min(irank, iwork2)
    iend   = ista + iwork1
    if iwork2 > irank:
        iend = iend + 1
    return int(ista),int(iend)

# %% date_related functions
def next_time(startdate,days=0,hours=0,minutes=0,seconds=0):
    """Find next time for calculation"""
    sdate = datetime.strptime(startdate, '%Y%m%d_%H')
    date = sdate + timedelta(days=days, hours=hours)
    date = datetime.strftime(date, '%Y%m%d_%H')
    return date

def date_range(startdate,enddate):
    """Time duration in hours"""
    sdate = datetime.strptime(startdate, '%Y%m%d_%H')
    edate = datetime.strptime(enddate  , '%Y%m%d_%H')
#    delta = edate - sdate
#     if isinstance(delta, np.timedelta64):
#         return delta.astype(timedelta).total_hours() / 60.
    date_arr = pd.date_range(sdate,edate,freq='6H')
    dates    = date_arr.strftime('%Y%m%d_%H')
    return dates
#    return delta.days, delta.seconds

#%%
#fixed_dir = "/Users/daohoa/Desktop/my-notebook/LAGRANGIAN/TSCOMBO2017/"
fixed_dir = "/work/users/hoadnq/lazy_make_it_better/"
cdir = fixed_dir + "results/"
data_dir = fixed_dir + "data/"
lab_dir  = fixed_dir + "CLU_VORT/nc_files/"
plt_dir  = fixed_dir + "laziness/"
# %%
########## initialize MPI communications ###########
comm = MPI.COMM_WORLD
myrank = comm.Get_rank()
nprocs = comm.Get_size()

assert nprocs > 1
##############
# %%
sdate = "20170712_00"
edate = "20170712_12"

dates = date_range(sdate,edate)

ndate = len(dates)


# %%
### MPI
nrows = ndate
######### Loop over all trajectories in MPI #############
ista, iend = para_range(0,nrows,nprocs,myrank)
iprev      = myrank - 1
inext      = myrank + 1
#if myrank == 0: iprev = MPI_PROC_NULL
if myrank == 0: print(nrows)
print ("I am process %d of %d running from %d to %d" %(myrank, nprocs, ista, iend))
######

for icase in range(ista,iend):
    rv_date = dates[icase]
    ow_lag = function.ow_lag(rv_date)
    vort   = function.vort(rv_date)

    rlon,rlat,rlev = function.get_dims(rv_date)

    # label netCDF
    nclab = netCDF4.Dataset(lab_dir + "cc2d_%s.nc" % rv_date)

    #print(nclab.variables.keys())
    # 
    names = [s for s in nclab.variables.keys() if "labels" in s]
    exclude_range = np.concatenate((np.arange(0,3,0.1),np.arange(17,40,0.1)))
    print(exclude_range)
    exclude = ['labels_cc2d_%05.2f' %x for x in exclude_range]
    #new_names = names.remove(exclude)
    ex_names = [x for x in names if x not in exclude]

    nlev = ow_lag.shape[0]
    # 
    data = []
    clevs = np.linspace(-0.0007,0.0007,51)
    for varname in ex_names:
        label_3d = nclab.variables[varname][:]

        fmt_value = varname[-5:]; value = float(fmt_value)
        print(value)
        for ilev in range(nlev):
            label_2d = label_3d[ilev,:,:]
            owlag_2d = ow_lag[ilev,:,:]

            # get unique labels (exclude zero)
            unique_lab = np.unique(label_2d)
            unique_lab = unique_lab[unique_lab != 0]

            # create the mask
            if unique_lab.shape != 0:
                for label in unique_lab:
                    get_index = np.where(label_2d == label)
                    min_owlag = np.nanmin(owlag_2d[get_index])
                    if min_owlag > 0:
                        data.append([rlev[ilev],value,label,min_owlag])
                        mask = np.empty(owlag_2d.shape)
                        mask[get_index] = owlag_2d[get_index] 
                        # do the plot
                        fig = plt.figure(figsize = (14,16))
                        ax = fig.add_subplot(111,projection = ccrs.PlateCarree())
                        cf = ax.contourf(rlon[:],rlat[:],mask,
                                     clevs,
                                     cmap = cm.RdBu_r,
                                     extend = 'max')
                        cs = ax.contour(rlon[:],rlat[:],ow_lag[ilev,:,:],
                                     [-1,0,1],
                                     colors = 'red',linewidth = 1)
                    
                        cbar = fig.colorbar(cf,orientation='horizontal')
                        for axis in ['top','bottom','left','right']:
                            ax.spines[axis].set_linewidth(2)
                        ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
                        ax.add_feature(cfeature.LAKES.with_scale('10m'),
                                color = 'black', linewidths=0.05)
                        ax.add_feature(cfeature.BORDERS.with_scale('10m'),
                                linestyle='-',  color = 'black',
                                linewidths=0.05)

                        anchored_text = AnchoredText("%i hPa" % rlev[ilev], loc=2,
                                        prop=dict(fontweight="bold",fontsize=18))

                        ax.add_artist(anchored_text)
                        
                        plt.savefig(plt_dir + 'possible_label_%s_%s_%03d_%s.png' %(rlev[ilev],fmt_value,label,rv_date) , dpi = 300)
                        plt.close(fig)
    # Create a pandas df
    df = pd.DataFrame(data,columns = ["level","value","label","min_owlag"],index=None)
    df.to_csv(plt_dir + 'possible_label_%s.csv' %rv_date, index = None)

    # Flush
    levels,lons,lats = 3*[None]; df,data = 2*[None]

MPI.Finalize()

# %%
#clevs = np.linspace(-0.0007,0.0007,51)
#for ilev in range(nlev):
#    fig = plt.figure(figsize = (14,16))
#    ax = fig.add_subplot(111)
#    cf = ax.contourf(ow_lag[ilev,:,:],
#                clevs,
#                 cmap = cm.RdBu_r,
#                 extend = 'max'
#                )
    #cs = 

#    cbar = fig.colorbar(cf,orientation='horizontal')

# %%
