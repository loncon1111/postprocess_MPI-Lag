#!/usr/bin/env python

#%% import libs 
import numpy as np
import pandas as pd
import math


def scaling(var,itime=None):
    if itime is None:
        sub = np.mean(var,axis=1)
        std = np.std(var,axis=1)
    else:
        sub = var[:,itime]
        std = np.std(var[:,itime])
    var = (var - sub[:,None])/std
    return var

def rotate(lon,lat,angle):
    new_lon = lon* math.cos(angle) - lat* math.sin(angle)
    new_lat = lon* math.sin(angle) + lat* math.cos(angle)
    return new_lon,new_lat

def get_head(lon0,lat0,lon1,lat1):
    dlon = lon1 - lon0
    if dlon > 180:
        dlon = dlon - 360.
    if dlon < -180:
        dlon = dlon + 360.
    dx = dlon * np.cos(np.deg2rad(lat0+lat1))
    dy = lat1 - lat0
    vec1 = (1.,0.)
    vec2 = (dx,dy)
    unit_vec1 = vec1 / np.linalg.norm(vec1)
    unit_vec2 = vec2 / np.linalg.norm(vec2)
    cos_product = np.dot(unit_vec1,unit_vec2)
    sin_product = np.cross(unit_vec1,unit_vec2)
    angle = np.arctan2(sin_product,cos_product)
    print(angle)
    #angle = np.where(angle < 0,math.pi + angle)
    return angle


