# coding:utf-8

# import libs
import numpy as np
import pandas as pd
import similaritymeasures
from sklearn.cluster import DBSCAN

def _get_3d_trajectories(traj_array):
    """
    """
    ntra, ntime = traj_array.shape
    trajectories = np.empty((ntra,ntime,3))

    trajectories[:,:,0] = traj_array['lon']
    trajectories[:,:,1] = traj_array['lat']
    trajectories[:,:,2] = traj_array['p']
    return trajectories 

def _get_2d_trajectories(traj_array):
    """
    """
    ntra, ntime = traj_array.shape
    trajectories = np.empty((ntra,ntime,2))

    trajectories[:,:,0] = traj_array['lon']
    trajectories[:,:,1] = traj_array['lat']
    return trajectories



# calculate distance matrix with order of N**2
def _distance_matrix(trajectories, method = "Frechet"):
    """
      param method: "Frechet" or "Area"
    """
    n = len(trajectories)
    dst_matrix = np.zeros((n,n))

    for i in range(n - 1):
        p = trajectories[i]
        for j in range(i + 1, n):
            q = trajectories[j]
            if method == "Frechet":
                dst_matrix[i,j] = similaritymeasures.frechet_dist(p,q)
            else:
                dst_matrix[i,j] = similaritymeasures.area_between_two_curves(p,q)
            dst_matrix[j,i] = dst_matrix[i,j]
    return dst_matrix

# DBSCAN clustering
def _cluster_by_dbscan(distance_matrix, eps = 1000):
    """ note that the trajectories are originally on lat/lon, not meter
     param eps: unit m for Frechet distance, m^2 for Area
    """
    cl = DBSCAN(eps = eps, min_samples = 1, metric = 'precomputed')
    # min_samples: 1, it is to make sure all trajectories will be clustered into a cluster
    cl.fit(distance_matrix)
    return cl.labels_ 
     

