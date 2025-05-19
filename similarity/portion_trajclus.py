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
import sys, getopt
from matplotlib.offsetbox import AnchoredText 

# %%
date = ''
try:
   opts, args = getopt.getopt(sys.argv[1:],"hi:", "input_date=")
except getopt.GetoptError:
   print('traject_clustering.py -i <input_date>')
   sys.exit(2)

for opt, arg in opts:
   if opt == '-h':
       print('traject_clustering.py -i <input_date>')
       sys.exit()
   elif opt in ("-i", "--input_date"):
       date = arg

#print('input_date is :', date)


# %%
cyclone = "SONCA"
#date    = "20170726_12"
#date    = "20170715_00"
# read distance matrix
#work_dir = "/Users/daohoa/Desktop/my-notebook/LAGRANGIAN/TSCOMBO2017/"
work_dir = "/work/users/hoadnq/lazy_make_it_better/"
dst_file = glob.glob(work_dir + 'similarity/distma/dist_matrix_%s_%s.*' %(cyclone,date))[0]
print(dst_file)
cyc_dir = work_dir + "vortex_regions/"
plt_dir = work_dir + "plot/"
datdir = work_dir + "data/ERA5/"

# %% get prefix
prefix = dst_file.split('.')[-1]
if prefix == 'out':
    dist_matrix = np.loadtxt(dst_file,delimiter=',')
if prefix == 'nc':
    nc_dist = xr.open_dataset(dst_file)
    print(nc_dist)
    dist_matrix = nc_dist['dist_matrix'].values

ntraj = np.shape(dist_matrix)[0]

#dist_matrix = dist_matrix.replace([np.inf, -np.inf], np.nan)
dist_matrix[dist_matrix == np.inf] = 10e21

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
ntra, ntime = traj_array.shape
# %% get boundaries
#minlon = np.nanmin(traj_array['lon'])
#maxlon = np.nanmax(traj_array['lon'])
#minlat = np.nanmin(traj_array['lat'])
#maxlat = np.nanmax(traj_array['lat'])
minlon = 90
maxlon = 150
minlat = 0
maxlat = 30

# %% select traj to plot
assert ntra == len(labels)

# %%



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

for itra in range(ntra):
    x = traj_array['lon'][itra]
    y = traj_array['lat'][itra]
    icol = np.squeeze(np.where(unique == labels[itra]))
    ax.plot(x,y,color = colors[icol],alpha = 0.5)
    #ax.scatter(x[0],y[0],color = colors[icol], s = 12, alpha = 0.5,marker = 11)
    #ax.scatter(x[-1],y[-1],color = colors[icol], s = 12, alpha = 0.5,marker = 11)

#for label in unique:

    #cind = np.squeeze(np.where(unique == labels[itra]))
    #print(traj_array['lon'][itra],traj_array['lat'][itra])
    #ax.scatter(traj_array['lon'][itra][0],traj_array['lat'][itra][0])
    #ax.plot(traj_array['lon'][traj_group],traj_array['lat'][traj_group])
plt.savefig(plt_dir + 'segments_%s_%s.pdf' %(cyclone,date))
#plt.savefig(plt_dir + 'portion_%s_%s.pdf' %(cyclone,date))
plt.close()
#exit()

# %% make a mask
all_levs = np.unique(traj_array['p'][:,0])
era_file = glob.glob(datdir+"P%s" %date)[0]
data = xr.open_dataset(era_file)
var = data['U'].sel(level = all_levs,
            longitude = slice(minlon,maxlon),
            latitude = slice(minlat,maxlat)).squeeze()

#data
mask = func_traj._init_mask(var)
mask = func_traj._get_mask(
    ncname = traj_file,
    mask_array = mask,
    labels = labels 
)
#mask = mask.where(mask != 0)
print(mask)

#plt.imshow(mask)
print(all_levs)

# %%
# plot masking
nlev = all_levs.size
columns = 2
rows = nlev // columns
rows += nlev % columns
fig = plt.figure(figsize = (12,12))

position = range(1,nlev + 1)

for irow, lev in enumerate(all_levs):
    print(rows,irow,position[irow])
    ax = fig.add_subplot(rows, columns, position[irow],
                         projection = ccrs.PlateCarree())

    ax.set_ylim(minlat,maxlat)
    ax.set_xlim(minlon,maxlon)
    #ax.add_feature(cfeature.LAND.with_scale('10m'))
    ax.add_feature(cfeature.BORDERS.with_scale('10m'))
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'))

    cs = plt.contourf(mask.longitude,mask.latitude,mask[irow,:,:],
                     levels = unique)

    ax.grid(linestyle = 'dotted', linewidth = 3, color = 'black')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)
    anchored_text = AnchoredText("%i hPa - %s - %s" %(lev,cyclone,date),
                                loc = 2,
                                prop = dict(fontweight='bold',fontsize = 13))
    ax.add_artist(anchored_text)
    


fig.tight_layout(h_pad = None, w_pad = None)
fig.subplots_adjust(hspace = 0,wspace = 0)
plt.savefig(plt_dir + 'portion_%s_%s.eps' %(cyclone,date))

exit()
