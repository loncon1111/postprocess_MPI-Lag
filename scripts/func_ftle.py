#/usr/bin/env python

#coding:utf-8

# import libs
import numpy as np

# Cauchy-Green strain tensor
def eig_cgStrain(xt,yt,xo,yo):
    ftlemat = np.zeros((2,2))
    ftlemat[0][0] = (xt[1]-xt[0])/(xo[1]-xo[0])
    ftlemat[1][0] = (yt[1]-yt[0])/(xo[1]-xo[0])
    ftlemat[0][1] = (xt[3]-xt[2])/(yo[3]-yo[2])
    ftlemat[1][1] = (yt[3]-yt[2])/(yo[3]-yo[2])
    if (True in np.isnan(ftlemat)): return 'nan'
    ftlemat = np.dot(ftlemat.transpose(), ftlemat)
    w, v = np.linalg.eigh(ftlemat)

    return w,v

# calculate FTLE field
def calc_ftle(traj_2x,traj_2y,xx,yy,nx,ny):
    ftle = np.zeros((nx,ny))
    for i in range(0,nx):
        for j in range(0,ny):
            # index 0:left, 1:right, 2:down, 3:up
            xt = np.zeros(4); yt = np.zeros(4)
            xo = np.zeros(4); yo = np.zeros(4)

            # central differencing except end points
            if i==0:
                xt[0] = traj_2x[i][j]; xt[1] = traj_2x[i+1][j]
                yt[0] = traj_2y[i][j]; yt[1] = traj_2y[i+1][j]
                xo[0] = xx[i][j]; xo[1] = xx[i+1][j]
            elif i==nx-1:
                xt[0] = traj_2x[i-1][j]; xt[1] = traj_2x[i][j]
                yt[0] = traj_2y[i-1][j]; yt[1] = traj_2y[i][j]
                xo[0] = xx[i-1][j]; xo[1] = xx[i][j]
            else:
                xt[0] = traj_2x[i-1][j]; xt[1] = traj_2x[i+1][j]
                yt[0] = traj_2y[i-1][j]; yt[1] = traj_2y[i+1][j]
                xo[0] = xx[i-1][j]
                xo[1] = xx[i+1][j]

            if j==0:
                xt[2] = traj_2x[i][j]; xt[3] = traj_2x[i][j+1]
                yt[2] = traj_2y[i][j]; yt[3] = traj_2y[i][j+1]
                yo[2] = yy[i][j]; yo[3] = yy[i][j+1]
            elif j==ny-1:
                xt[2] = traj_2x[i][j-1]; xt[3] = traj_2x[i][j]
                yt[2] = traj_2y[i][j-1]; yt[3] = traj_2y[i][j]
                yo[2] = yy[i][j-1]; yo[3] = yy[i][j]
            else:
                xt[2] = traj_2x[i][j-1]; xt[3] = traj_2x[i][j+1]
                yt[2] = traj_2y[i][j-1]; yt[3] = traj_2y[i][j+1]
                yo[2] = yy[i][j-1]; yo[3] = yy[i][j+1]
    
            lambdas,eig_vect = eig_cgStrain(xt, yt, xo, yo)
            if (lambdas=='nan'):
                ftle[i][j] = float('nan')
            else:
                ftle[i][j] = .5*np.log(max(lambdas))/(12*3600)
    return ftle


