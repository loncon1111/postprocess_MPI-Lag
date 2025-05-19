#!/usr/bin/env python

#%% import libs
#%% import libs
import numpy as np
import pandas as pd
import xarray as xr
import class_plot_the as traj_func
import function_plotting as func_plot
import glob, sys, math
from datetime import datetime,timedelta

from collections import ChainMap

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import netCDF4
from matplotlib.offsetbox import AnchoredText

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib as mpl
from labellines import labelLine, labelLines


if __name__ == "__main__":
    #cyclone = "HAITANG"; date = "20170728_00"; traj = 1266
    cyclone = "ROKE"; date = "20170721_06"; traj = 1129
    #cyclone = "SONCA"; date = "20170720_18"; traj = 409
    #cyclone = "NESAT"; date    = "20170725_18";traj = 1075
    #cyclone = "TALAS"; date = "20170715_00"; traj = 460
    # read distance matrix
    work_dir = "/work/users/hoadnq/lazy_make_it_better/"
    cyc_dir = work_dir + "vortex_regions/"
    plt_dir = work_dir + "plot/"
    datdir = work_dir + "data/ERA5/"
    trmm_dir = work_dir + "TRMM/" 
    # %%
    # read trajectories
    traj_file = cyc_dir + "lsl_%s_%s_clt.4" %(date,cyclone)
    traj_array = traj_func._traj_array(traj_file)
    # %%
    df = pd.read_excel(work_dir + "72h_bwd_clus_%s_%s.xlsx" %(date,cyclone))
    labels = df['labels'].to_numpy()

    cat_df = pd.read_excel(work_dir + 'scripts/72h_bwd_cat_wes.xlsx')
    unique = np.unique(labels)
    wes_cat = cat_df[cyclone].to_numpy()
    wrap_cat = np.setxor1d(unique,wes_cat)
    labs = ChainMap(dict.fromkeys(wrap_cat,0),
                    dict.fromkeys(wes_cat,1)) #haitang
    cat = df['labels'].map(labs.get).to_numpy(dtype = np.int32)
    # %%
    traj_array['pv'] = traj_array['pv'] *1e6
    #%%
    era_ds = xr.open_dataset(datdir+"P%s" %date)
    # %%
    
    # %% get boundaries
    minlon = 120
    maxlon = 140
    minlat = 0
    maxlat = 30
    #%%
    conv_date = datetime.strptime(date,"%Y%m%d_%H")
    ntra,ntime = traj_array.shape
    time = ntime - 1
    #%%
    typ_trarr = traj_array[traj]
    cyc_days = pd.date_range(start=conv_date,end=conv_date-timedelta(hours=time),freq='-1H')
    days = pd.date_range(start=cyc_days[0],end=cyc_days[-1],freq='-1D')
    print(days)
    date_range = pd.date_range(start=days[0],end=days[-1],freq='-1D',normalize=True).strftime("%Y%m%d")
    print(date_range)
    trmm_fils = [trmm_dir + '3B-HHR.MS.MRG.3IMERG.%s.V06B.HDF5.nc4' %(x) for x in date_range]
    trmm_ds = xr.open_mfdataset(trmm_fils).sel(lon=slice(minlon,maxlon),lat=slice(minlat,maxlat)).squeeze()
    prec = trmm_ds["precipitationCal"]
    del trmm_ds

    #%%

    fig = plt.figure(figsize = (12,9))
    mpl.rcParams.update({'font.size':16})
    ax = fig.add_subplot(111,projection = ccrs.PlateCarree())

    ax.set_xlim(int(minlon), int(maxlon))
    ax.set_ylim(int(minlat), int(maxlat))
    # add geographical features
    ax.add_feature(cfeature.BORDERS.with_scale('10m'),
                linestyle = '-',
                linewidths = 0.5,zorder=25)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'),
                linestyle = '-',
                linewidths = 0.8,zorder=25)
    
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)

    clevs = [5,10,20,30,40,50,60,70,80,90,100]

    for iday, day in enumerate(np.delete(days,-1)):
        print(day)
        prvday = day - timedelta(hours=23,minutes=59)
        prc = prec.sel(time=slice(prvday,day)).sum(dim='time')
        prc = xr.where(prc < 0.1,np.nan,prc)
        itim = np.squeeze(np.where(cyc_days==day))
        iptim = np.squeeze(np.where(cyc_days==(day-timedelta(days=1))))
        mnlon = typ_trarr["lon"][iptim]; mxlon = typ_trarr["lon"][itim]
        mnlat = typ_trarr["lat"][iptim] - 10; mxlat = typ_trarr["lat"][itim]+10
        print(mnlon,mxlon)
        try:
            assert mnlon < mxlon
        except AssertionError:
            a = mnlon; mnlon = mxlon; mxlon = a; del a

        prc = prc.sel(lon=slice(mnlon,mxlon),lat=slice(mnlat,mxlat))
        print(prc)

        # open era5
        era5_ds = xr.open_dataset(datdir + 'P%s' %day.strftime("%Y%m%d_%H"))
        print(era5_ds)
        u10 = era5_ds['u10'].sel(longitude=slice(mnlon,mxlon),latitude=slice(mnlat,mxlat)).squeeze()
        v10 = era5_ds['v10'].sel(longitude=slice(mnlon,mxlon),latitude=slice(mnlat,mxlat)).squeeze()
        print(u10)
        skip = 5
        qv = ax.quiver(x=u10.longitude[::skip],y=u10.latitude[::skip],u=u10[::skip,::skip],v=v10[::skip,::skip],color='black',pivot='mid',scale_units='inches',scale = 25,zorder = 40)

        qvk = ax.quiverkey(qv, 0.9,0.9,1,r'$25 \frac{m}{s}$', coordinates='figure',zorder=40)

        cs = ax.contourf(prc.lon,prc.lat,prc.T,levels=clevs,
                    cmap = 'bone_r',zorder =10,extend='max',vmin=5,
                    alpha = 1)
        ax.add_patch(Rectangle((mnlon,mnlat),mxlon-mnlon,mxlat-mnlat,
                    edgecolor = 'black',
                    facecolor = None,
                    fill = False,
                    lw = 2,
                    zorder = 30))
        ax.contour(prc.lon,prc.lat,prc.T,levels=[50],
                    colors='black',zorder =40)


    trajs = np.squeeze(np.where(cat == 1))
    if cyclone == "TALAS":
        traj_array["pv"] = - traj_array["pv"]

    cf = func_plot.plot_trajs(
        ax = ax,
        trajs = traj_array[trajs],
        variable = 'pv',
        levels = np.linspace(-0.4,0.4,11),
        zorder = 20
    )
    fig.colorbar(cf,orientation='horizontal')
    fig.colorbar(cs,orientation='horizontal')
    
    plt.savefig('pvw_env_%s_%s.pdf' %(date,cyclone))


 
##%%
    




#
