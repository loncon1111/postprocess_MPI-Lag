#/usr/env python

#coding:utf-8

# import libs
import numpy as np
import xarray as xr
import netCDF4
import glob
import pandas as pd
import class_plot_the as func_traj
import similaritymeasures
import function_ml
from function_mpi import para_range
from mpi4py import MPI

########## initialize MPI communications ###########
comm = MPI.COMM_WORLD
myrank = comm.Get_rank()
nprocs = comm.Get_size()

assert nprocs > 1


# 
workdir = "/work/users/hoadnq/lazy_make_it_better/"
traj_dir = workdir + "vortex_regions/"

cyc_name = "TALAS"
date = "20170715_00"

lsl_file = glob.glob(traj_dir + "lsl_%s_%s_clt.4" %(date,cyc_name))[0]

# read trajectories
traj_array = func_traj._traj_array(lsl_file)

level = 850

traj = traj_array[traj_array['p'][:,0] == level]
print(traj.shape)
ntraj, ntime = traj.shape
print(len(traj))

# get the distance matrix
trajectories = function_ml._get_trajectories(traj)

n = len(trajectories)
dst_matrix = np.zeros((n,n))
method = "Frechet"

######### Loop over all trajectories in MPI #############
ista, iend = para_range(0,n-1,nprocs,myrank)
iprev      = myrank - 1
inext      = myrank + 1
if myrank == 0: print(n)
print ("I am process %d of %d running from %d to %d" %(myrank, nprocs, ista, iend))

t_start = MPI.Wtime()

for i in range(ista,iend):
    p = trajectories[i]
    for j in range(i + 1, n):
        q = trajectories[j]
        if method == "Frechet":
           dst_matrix[i,j] = similaritymeasures.frechet_dist(p,q)
        else:
           dst_matrix[i,j] = similaritymeasures.area_between_two_curves(p,q)
        dst_matrix[j,i] = dst_matrix[i,j]

t_diff = MPI.Wtime() - t_start

print( "Process %d finished in %5.4fs.\m" %(myrank,t_diff))

####### Send and receive in MPI ##########
# intitiate rank communication for MPI
ureq  = np.empty(nprocs,dtype = MPI.Request)
tags  = np.arange(nprocs)

if myrank == 0:
    data = np.zeros((n,n),dtype = np.float64)
    data[ista:iend,ista+1:n ] = dst_matrix[ista:iend,ista+1:n]
    data[ista+1:n ,ista:iend] = dst_matrix[ista+1:n ,ista:iend]

    # gather updates from all cores (non-contiguous memory)
    for irank in range(1,nprocs):
        jsta, jend = para_range(0,nrows,nprocs,irank)
        comm.Recv([dst_matrix,MPI.DOUBLE], source = irank, tag = irank)

        print("Received info from ",irank)
        data[jsta:jend,jsta+1:n] = dst_matrix[jsta:jend,jsta+1:n]
        data[jsta+1:n,jsta:jend] = dst_matrix[jsta+1:n,jsta:jend]

else:
    comm.Send([dst_matrix,MPI.DOUBLE], dest = 0, tag = myrank)    

comm.Barrier()
#dist_m = function_ml._distance_matrix(traj_locs, method = "Frechet")

print(data)

# perform trajectory clustering
labels = function_ml._cluster_by_dbscan(data, eps = 1000)

print(labels)
print(labels.shape)

np.savetxt('clustering.test.out', labels, delimiter = ',')

 



