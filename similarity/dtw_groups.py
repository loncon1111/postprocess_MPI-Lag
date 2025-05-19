#/usr/env python 
#%% import libs
import sys, glob, getopt
import matplotlib
import numpy as np
np.set_printoptions(threshold = sys.maxsize)

from sklearn.cluster import DBSCAN
import xarray as xr
import class_plot_the as func_traj
import netCDF4
from shapely.geometry import Point,LineString
from sklearn.cluster import AgglomerativeClustering
import function_ml
#from dtaidistance import dtw_visualization as dtwvis
from dtaidistance import dtw_ndim
from dtaidistance import dtw
from scipy.spatial import distance_matrix
from itertools import product
import pandas as pd
# %%
cyclone = "HAITANG"
typical_trajs = [1266,325]
date    = "20170728_00"
dthres = 8e7
# read distance matrix
work_dir = "/work/users/hoadnq/lazy_make_it_better/"
dst_file = glob.glob('./trial_agglo/dist_matrix_%s_%s.*' %(cyclone,date))[0]
cyc_dir = work_dir + "vortex_regions/"



# %% read trajectories
# read trajectories
lsl_file = glob.glob(cyc_dir + "lsl_%s_%s_clt.4" %(date,cyclone))[0]
traj_array = func_traj._traj_array(lsl_file)
# %%
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
cl = AgglomerativeClustering(
     n_clusters=None,
     affinity='precomputed',
     linkage = 'complete',
     distance_threshold = 8e7
)
cl.fit(dist_matrix)
labels = cl.labels_

# %%
# %% get boundaries
minlon = 90
maxlon = 150
minlat = 0
maxlat = 30
cl1 = AgglomerativeClustering(
    n_clusters = None,
    affinity='precomputed',
    linkage = 'complete',
    distance_threshold = 2e7        
    )
# %%

#start = xxxx
#end   = start + 20

trajectories = function_ml._get_2d_trajectories(traj_array)
for itra in typical_trajs:
    label = labels[itra]
    lab_group = np.squeeze(np.where(labels == label))
    #lab_group = np.setdiff1d(np.where(labels == label),np.array(itra))
    #sel_list = list(product(lab_group,lab_group))
    #new_dist_ma = dist_matrix[lab_group][:,lab_group]
    #lev = np.squeeze(traj_array["p"][lab_group,0])
    #lower = np.squeeze(lab_group[np.argwhere(np.logical_and(lev <= 850, lev >= 700))])

    #### DTW
    s1 = trajectories[itra,:,:]
    for iquery,query in enumerate(lab_group):
        s2 = trajectories[query,:,:]
        d, paths = dtw_ndim.warping_paths(s1, s2)
        path = np.asarray(dtw.best_path(paths))


        df = pd.DataFrame(path, columns = ["time", str(query)])
        df.sort_values(by = ['time'], inplace = True)
        first_72 = df[df[str(query)] == 72.][str(query)].idxmax()
        df[str(query)].where(df[str(query)] != 72., np.nan, inplace = True) 
        df[str(query)].iloc[first_72] = 72.
   
        if iquery == 0:
            df1 = df
        else:
            df["sequence"] = df.groupby('time').cumcount()
            df1["sequence"] = df1.groupby('time').cumcount()

            df1 = pd.merge(df1,df, on = ['time','sequence'], how = 'outer').drop(labels='sequence',axis=1)
            #df1 = df1.combine_first(df)
            #df1 = df1.merge(df, how = "l" left_index = True, right_index = True)
        df = None

    df1.sort_values(by = ['time'], inplace = True)

    excel_name = cyclone + "_" + date + "_dtw_%s.xlsx" %itra
    df1.to_excel(excel_name, index = False)
    # Flush
    df1 = None

