#/usr/env python

#%% import libs
import sys
import numpy as np

np.set_printoptions(threshold=sys.maxsize)

from sklearn.cluster import DBSCAN
import xarray as xr
import glob
import matplotlib.pyplot as plt
import class_plot_the as func_traj
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import netCDF4

# %%
cyclone = "HAITANG"
date    = "20170726_12"
#date    = "20170727_06"
# read distance matrix
#work_dir = "/Users/daohoa/Desktop/my-notebook/LAGRANGIAN/TSCOMBO2017/"
work_dir = "/work/users/hoadnq/lazy_make_it_better/"
dst_file = glob.glob(work_dir + 'similarity/distma/dist_matrix_%s_%s.*' %(cyclone,date))[0]
cyc_dir = work_dir + "vortex_regions/"
plt_dir = work_dir + "plot/"

# %% get prefix
prefix = dst_file.split('.')[-1]
if prefix == 'out':
    dist_matrix = np.loadtxt(dst_file,delimiter=',')
if prefix == 'nc':
    nc_dist = xr.open_dataset(dst_file)
    print(nc_dist)
    dist_matrix = nc_dist['dist_matrix'].values

ntraj = np.shape(dist_matrix)[0]


#%%
epsilon = 1.
# Apply DBSCAN
cl = DBSCAN(eps = epsilon, min_samples = 1, metric = 'precomputed')
cl.fit(dist_matrix)

labels = cl.labels_
#%%
# grep
unique = np.unique(labels)
print(unique) 
# make the colormap
ncols = len(unique)
colormap = plt.get_cmap('viridis',ncols)
colors = colormap(np.linspace(0,1,ncols))

# %% read trajectories
# read trajectories
traj_file = cyc_dir + "lsl_%s_%s_clt.4" %(date,cyclone)
traj_array = func_traj._traj_array(traj_file)
# %%
traj_array['lat']
#%%
ncfile = netCDF4.Dataset(traj_file)
ntra, ntime = func_traj._get_netcdf_traj_dim(ncfile)
# %% get boundaries
minlon = np.nanmin(traj_array['lon'])
maxlon = np.nanmax(traj_array['lon'])
minlat = np.nanmin(traj_array['lat'])
maxlat = np.nanmax(traj_array['lat'])

# %% select traj to plot
assert ntra == len(labels)

# %%
for ilabel,label in enumerate(unique):
    traj_group = np.squeeze(np.where(labels == label))

    print(traj_group)


    fig = plt.figure(figsize = (12,9))
    ax = fig.add_subplot(111,projection = ccrs.PlateCarree())

    ax.set_xlim(int(minlon) - 2, int(maxlon) + 2)
    ax.set_ylim(int(minlat) - 2, int(maxlat) + 2)
    # add geographical features
    ax.add_feature(cfeature.BORDERS.with_scale('10m'),
                linestyle = '-',
                linewidths = 0.5)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'),
                linestyle = '-',
                linewidths = 0.8)

    #for itra in traj_group:
    #    cind = np.squeeze(np.where(unique == labels[itra]))
    #    ax.plot(traj_array['lon'][itra],traj_array['lat'][itra],
    #        color = colors[cind])
    ax.plot(traj_array['lon'][traj_group],traj_array['lat'][traj_group])

    plt.savefig(plt_dir + 'traject_%04d_%s_%s.png' %(label,date,cyclone), dpi = 600)
    plt.close()

    fig = plt.figure(figsize = (12,9))
    ax = fig.add_subplot(111)
    ax.set_xlim(0,72)
    ax.set_ylim(1000,200)
    plt.gca().invert_yaxis()
    ax.plot(range(0,73),traj_array["p"][traj_group].T)
    plt.savefig(plt_dir + 'p_%04d_%s_%s.png' %(label,date,cyclone), dpi = 600)





