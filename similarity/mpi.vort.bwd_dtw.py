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
import function_diagma
from function_mpi import para_range
from function_scale import scaling,rotate,get_head
from mpi4py import MPI



########## initialize MPI communications ###########
comm = MPI.COMM_WORLD
myrank = comm.Get_rank()
nprocs = comm.Get_size()

assert nprocs > 1


# 
workdir = "/work/users/hoadnq/lazy_make_it_better/"
#workdir = "/work/users/hmo/truongnm/MPI_LCS/lazy_make_it_better/"
traj_dir = workdir + "vortex_regions/"

cyc_name = "HAITANG"; date = "20170728_12"
#cyc_name =  "TALAS"; date = "20170715_00"
#cyc_name = "SONCA"; date = "20170720_18"
#cyc_name = "ROKE"; date = "20170721_18"
#cyc_name = "NESAT"; date = "20170725_18"

lsl_file = glob.glob(traj_dir + "lsl_%s_%s_clt.4" %(date,cyc_name))[0]
#lsl_file = traj_dir + "vortex_%s_%s_clt.4" %(date,cyc_name)

# read trajectories
traj_array = func_traj._traj_array(lsl_file)
traj_array = traj_array[:,::-1]

lon = traj_array['lon']; lat = traj_array['lat']; p = traj_array['p']
itime = 60
ntra, ntime = np.shape(traj_array)
head = [get_head(lon[x,itime],lat[x,itime],lon[x,itime+1],lat[x,itime+1]) for x in range(ntra)]
head = np.array(head)

#traj_array['lon'] = scaling(lon,itime)
#traj_array['lat'] = scaling(lat,itime)
lon = scaling(lon,itime)
lat = scaling(lat,itime)

traj_array['p']   = scaling(p  ,itime)

#for itra in range(ntra):
#    lon[itra],lat[itra] = rotate(lon[itra],lat[itra],head[itra])

#traj_array['lon'] = lon
#traj_array['lat'] = lat

#level = 850
all_levs = [850,800,750,700,650,600,500]

#cut off the lastest 6 hours
traj = traj_array[:,0:60]
#traj = traj_array[traj_array['p'][:,0] == level for level in all_levs]
ntraj, ntime = traj.shape

# get the distance matrix
trajectories = function_ml._get_2d_trajectories(traj)

n = int(len(trajectories))
method = "DTW"

# triangular upper 
ntotal = int(n*(n-1)/2)
triu_data = np.zeros(ntotal)

######### Loop over all trajectories in MPI #############
ista, iend = para_range(0,ntotal,nprocs,myrank)
iprev      = myrank - 1
inext      = myrank + 1
if myrank == 0: print(n)
print ("I am process %d of %d running from %d to %d" %(myrank, nprocs, ista, iend))

t_start = MPI.Wtime()

for index in range(ista,iend):
    i, j = function_diagma.linear_to_ij(n,index)
    p    = trajectories[i,:,:]
    q    = trajectories[j,:,:]

    if method == "Frechet":
        triu_data[index] = similaritymeasures.frechet_dist(p,q)
    if method == "area":
        triu_data[index] = similaritymeasures.area_between_two_curves(p,q)
    if method == "DTW":
        triu_data[index],d = similaritymeasures.dtw(p,q)
t_diff = MPI.Wtime() - t_start

print( "Process %d finished in %5.4fs.\m" %(myrank,t_diff))

####### Send and receive in MPI ##########
# intitiate rank communication for MPI
ureq  = np.empty(nprocs,dtype = MPI.Request)
tags  = np.arange(nprocs)

if myrank == 0:
    triu_dist = np.zeros((ntotal),dtype = np.float64)  # store upper triangular linear matrix

    # store taken data from master rank
    triu_dist[ista:iend] = triu_data[ista:iend]
    
    # gather updates from all cores (non-contiguous memory)
    for irank in range(1,nprocs):
        jsta, jend = para_range(0,ntotal,nprocs,irank)
        comm.Recv([triu_data, MPI.DOUBLE], source = irank, tag = irank)
        print("Received in info from ",irank)

        triu_dist[jsta:jend] = triu_data[jsta:jend]
else:
    comm.Send([triu_data,MPI.DOUBLE], dest = 0, tag = myrank)

comm.Barrier()

# now only rank 0 handle this
if myrank == 0:
    # put it into diagonal matrix
    dst_matrix = np.zeros((n,n))  # this distance matrix is triangular

    dst_matrix[np.triu_indices_from(dst_matrix, k = 1)] = triu_dist
    dst_matrix = dst_matrix + dst_matrix.T
    #np.savetxt('dist_matrix_%s_%s.out' %(cyc_name,date), dst_matrix, delimiter = ',')
    # perform trajectory clustering
    #labels = function_ml._cluster_by_dbscan(dst_matrix, eps = 0.5)

    # create xarray of distance matrix
    dist = xr.DataArray(
        data = dst_matrix,
        dims = ['x','y'],
        coords = dict(
            x = (["x"], range(n)),
            y = (["y"], range(n)),
        ),
        attrs = dict(
            description = "Distance Matrix",
            units = "deg"
        ),
        name  = 'dist_matrix',
    )

    print("...to file...")
    dist_file = "dist_matrix_cutbwd_2d_%s_%s.nc" %(cyc_name,date)
    dist.to_netcdf(dist_file,"w")

    #print(labels)
    #print(labels.shape)

    #np.savetxt('clustering.%s.%d.test.out'  %(cyc_name,date) , labels, delimiter = ',')

 



