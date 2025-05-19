#coding=utf-8

#%%
import netCDF4
import numpy as np
import pandas as pd
from datetime import datetime,timedelta
import matplotlib.pyplot as plt

#%%
def next_time(startdate,days=0,hours=0,minutes=0,seconds=0):
    """
    Find next time for calculation
    """
    sdate = datetime.strptime(startdate, '%Y%m%d_%H')
    date  = sdate + timedelta(days=days, hours=hours)
    date  = datetime.strftime(date, "%Y%m%d_%H")
    return date

def duration(startdate,enddate):
    """
    Time duration in hours
    """
    sdate = datetime.strptime(startdate, "%Y%m%d_%H")
    edate = datetime.strptime(enddate,   "%Y%m%d_%H")
    delta = edate - sdate
    return delta.days, delta.seconds

def connect_label(min_lon,max_lon,min_lat,max_lat, \
                  min_rlon,max_rlon,min_rlat,max_rlat, \
                  rflag):
    if min_lon > min_rlon and \
       min_lat > min_rlat and \
       max_lon < max_rlon and \
       max_lat < max_rlat:
        flag = rflag
    else:
        flag = 0
    return flag


#%%
cdir = "/Users/daohoa/Desktop/my-notebook/LAGRANGIAN/TSCOMBO2017/results"
ncdir = "/Users/daohoa/Desktop/my-notebook/LAGRANGIAN/TSCOMBO2017/CLU_VORT/nc_files"
# %%
sdate = "20170715_00"
edate = "20170715_00"

dur_day, dur_sec = duration(sdate,edate)
dur_hr = dur_sec // (60*60) + dur_day * 24
ndate = dur_hr // 6 # time interval

# %%
rv_date = sdate
level   = 650

print(ndate)


# %%
for icase in range(ndate+1):
    df = pd.read_csv(cdir + "/label_list2d_%s_%s.csv" %(rv_date,level),index_col=0)
    df["hull_minlon"] = np.nan
    df["hull_maxlon"] = np.nan
    df["hull_minlat"] = np.nan
    df["hull_maxlat"] = np.nan
    df["owlag_flag"]  = np.nan
    # Read netCDF cluster file
    f = netCDF4.Dataset(ncdir + "/cc2d_%s.nc" %rv_date,"r")
    lons = f.variables["longitude"][:]
    lats = f.variables["latitude"][:]
    levs = f.variables["level"][:]

    nlon = len(lons)
    nlat = len(lats)
    nlev = len(levs)

    ilev = np.squeeze(np.where(levs == level))

    nrows = len(df)

    for irow in range(nrows):
        label = df["label"].iloc[irow]
        value = df["value"].iloc[irow]

        # Get variable
        var = np.squeeze(f.variables["labels_cc2d_%05.2f" %value][ilev,:,:])

        latind, lonind = np.where(var == label)
        df["hull_minlon"].iloc[irow] = lons[min(lonind)]
        df["hull_minlat"].iloc[irow] = lats[min(latind)]
        df["hull_maxlon"].iloc[irow] = lons[max(lonind)]
        df["hull_maxlat"].iloc[irow] = lats[max(latind)]

        # Flush
        label = None; var = None; value = None
        latind,lonind = 2*[None]

    df["owlag_flag"] = [1 if check > 0 else 0 for check in df["min_owlag"]]


    df["hull_flag"] = 0

    exist_flag = 1
    for irow in range(nrows):
        ref_flag = df["hull_flag"].iloc[irow]

        #if ref_flag == 0:
        #    ref_flag = exist_flag    # initial a new flag if it does not exist

        # Flush
        #ref_flag = None
        if ref_flag == 0:
            ref_flag = exist_flag
            r_mnlon = df["hull_minlon"].iloc[irow]
            r_mnlat = df["hull_minlat"].iloc[irow]
            r_mxlon = df["hull_maxlon"].iloc[irow]
            r_mxlat = df["hull_maxlat"].iloc[irow]

            df["hull_flag"].iloc[irow] = ref_flag
            for jrow in range(irow,nrows):

                mnlon = df["hull_minlon"].iloc[jrow]
                mnlat = df["hull_minlat"].iloc[jrow]
                mxlon = df["hull_maxlon"].iloc[jrow]
                mxlat = df["hull_maxlat"].iloc[jrow]
                check_flag = connect_label(mnlon,mxlon,mnlat,mxlat, \
                 r_mnlon,r_mxlon,r_mnlat,r_mxlat, \
                 ref_flag)

                if check_flag != 0:
                    df["hull_flag"].iloc[jrow] = check_flag
                # Flush
                mnlon,mnlat,mxlon,mxlat,check_flag = 5*[None] 
            exist_flag = exist_flag + 1
            # Flush
            r_mnlon,r_mxlon,r_mxlat,r_mnlat = 4*[None]

    sorted_flag = df.groupby('hull_flag').count().sort_values(by = ["hull_centx"],
                                ascending = False).index

    #sub_df = df[df["hull_flag"] == sorted_flag[0]]
    #sub_df.to_csv(cdir + "/label_list2d_%s_%s_vort.csv" %(rv_date,level))

    df.to_csv(cdir + "/label_list2d_%s_%s.csv" %(rv_date,level))
    rv_date = next_time(startdate = rv_date, hours = 6)
    # Flush
    nrows,nlon,nlat,nlev = 4*[None]
    lons,lats,levs = 3*[None]; ilev = None

# %%
df
# %%
sorted_flag = df.groupby('hull_flag').count().sort_values(by = ["hull_centx"],
                                ascending = False).index
# %%
sorted_flag
# %%
sub_df = df[df["hull_flag"] == sorted_flag[0]]

# %%
sub_df
# %%
sub_df["owlag_flag"].to_numpy()
# %%
new_df = sub_df[['label','value']]
# %%
new_df["level"] = level
new_df
# %%
new_df['number_of_labels'] = new_df.groupby('value').count()['level'].to_numpy()
# %%
new_df.to_csv(cdir + '/labellist2d_%s_%s_vort.csv' %(rv_date,level) ,index = False)
# %%
sub_df.to_csv(cdir + "/label_list2d_%s_%s_vort.csv" %(rv_date,level))
# %%
