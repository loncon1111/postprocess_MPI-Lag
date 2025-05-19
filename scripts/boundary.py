#coding=utf-8

#%%
import netCDF4
import numpy as np
import pandas as pd
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridsprec
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredText
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os


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
# %%
#%%
cdir = "/Users/daohoa/Desktop/my-notebook/LAGRANGIAN/TSCOMBO2017/results/"
data_dir = "/Users/daohoa/Desktop/my-notebook/LAGRANGIAN/TSCOMBO2017/data/"

# %%
sdate = "20170715_00"
edate = "20170715_00"

dur_day, dur_sec = duration(sdate,edate)
dur_hr = dur_sec // (60*60) + dur_day * 24
ndate = dur_hr // 6 # time interval

# %%
rv_date = sdate
level   = 500
# %%
df = pd.read_csv(cdir + "label_list2d_%s_%s.csv" %(rv_date,level),index_col=0)

# %%
df
# %%
df = df[df["hull_centx"] != 0][df["hull_centy"] != 0]
# %%
df_frst = pd.DataFrame(columns = df.columns)
df_frst
# %%
hull_flag = df.groupby("hull_flag").first().index

# %%
hull_flag
# %%
for iflag, flag in enumerate(hull_flag):
    df_subflag = df[df["hull_flag"] == flag]
    #first_check = np.squeeze(np.where(df_subflag["owlag_flag"] == 0))
    #print(iflag, first_check)
    #if first_check.size > 1:
    #    first = first_check[-1] + 1
    #elif first_check.size == 1:
    #    first = first_check
    #else:
    #    first = 0
    first_check = df_subflag[df_subflag["owlag_flag"] == 1]
    print(df_subflag)
    first = first_check.index[0]

    print(df.iloc[first])

    #if first > df_subflag.index.size-1:
    #    first = None

    #if first != None:
    #    df_frst = df_frst.append(df_subflag.iloc[first])
    # Flush
    df_subflag = None; first_check , first = 2*[None]
# %%
df_frst
# %%
## Phase 2: Open netCDF file of label
ncdir = "/Users/daohoa/Desktop/my-notebook/LAGRANGIAN/TSCOMBO2017/CLU_VORT/nc_files/"
src = netCDF4.Dataset(ncdir + "cc2d_%s.nc" %rv_date)
# %%
nrows = df_frst.index.size

# %%
lon_lst = np.empty(0,dtype=int)
lat_lst = np.empty(0,dtype=int)
for irow in range(nrows):
    value = df_frst["value"].iloc[irow]
    label = df_frst["label"].iloc[irow]
    
    rlev  = src.variables["level"][:]
    rlon  = src.variables["longitude"][:]
    rlat  = src.variables["latitude"][:]

    ilev  = np.squeeze(np.where(rlev == level))

    var   = src.variables["labels_cc2d_%05.2f" %value][ilev,:,:]
    ndims = np.shape(var)
    lonind,latind = np.where(var == label)
    lon_lst = np.append(lon_lst,lonind)
    lat_lst = np.append(lat_lst,latind)
    # Flush
    #lonind,latind = 2*[None], value,label = 2*[None]

# %%
mask_frst = np.zeros(ndims,dtype=int)
# %%
mask_frst[lon_lst,lat_lst] = 1
# %%

## Phase 3: Plot masking array
x_tick_labels = [u'95\N{DEGREE SIGN}E', u'105\N{DEGREE SIGN}E',
                u'115\N{DEGREE SIGN}E', u'125\N{DEGREE SIGN}E',
                u'135\N{DEGREE SIGN}E', u'145\N{DEGREE SIGN}E']
y_tick_labels = [u'0\N{DEGREE SIGN}', u'5\N{DEGREE SIGN}N',
                u'10\N{DEGREE SIGN}N', u'15\N{DEGREE SIGN}N',
                u'20\N{DEGREE SIGN}N', u'25\N{DEGREE SIGN}N',
                u'30\N{DEGREE SIGN}N']

# %%
#mask_final = np.zeros(ndims,dtype=int)
#imask_x, imask_y = np.where(mask_frst == 1)
#for ipt in range(len(imask_x)):
#    if ow_lag[imask_x[ipt],imask_y[ipt]] > 0:
#        mask_final[imask_x[ipt],imask_y[ipt]] = 1
#    else:
#        mask_final[imask_x[ipt],imask_y[ipt]] = 0

# %%
## Phase 4: Plot only vertical vorticity
zeta_fw = netCDF4.Dataset(data_dir + "forward/" + "vort_%s.nc" %rv_date)
zeta_bw = netCDF4.Dataset(data_dir + "backward/" + "vort_%s.nc" %rv_date)
vort_fw = zeta_fw.variables["relvort"][ilev,:,:]
vort_bw = zeta_bw.variables["relvort"][ilev,:,:]
vort    = vort_fw + vort_bw

ow_fw = netCDF4.Dataset(data_dir + "forward/" + "new_%s.nc" %rv_date)
ow_bw = netCDF4.Dataset(data_dir + "backward/" + "new_%s.nc" %rv_date)
ieig1_fw = ow_fw.variables["ieig1"][ilev,:,:]
ieig1_bw = ow_bw.variables["ieig1"][ilev,:,:]
reig1_fw = ow_fw.variables["reig1"][ilev,:,:]
reig1_bw = ow_bw.variables["reig1"][ilev,:,:]

vort    = vort_fw + vort_bw

ieig1 = ieig1_fw + ieig1_bw
reig1 = reig1_fw + reig1_bw
ow_lag = ieig1 - reig1

mask_frst = mask_frst.astype(bool)
mask_vort = np.ma.masked_where(mask_frst == 0,vort)

# %%
import matplotlib
#font = 'tex gyre heros'  # an open type font
#matplotlib.rcParams['font.sans-serif'] = font
#matplotlib.rc('mathtext', fontset='custom', it=font + ':italic')
#matplotlib.rc('font', size=13)  # change font size from default 10


font = {'family': 'sans-serif',
        'color':  'black',
        'weight': 'bold',
        'size': 16,
        }

matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

clevs = np.arange(0,30,0.5)
clevs_ct = [-1,0,1]
ax = plt.axes(projection = ccrs.PlateCarree())

cf = ax.contourf(rlon[:],rlat[:],mask_vort*1e4, clevs, 
            cmap=plt.cm.Spectral_r,extend = 'both')  
cs = ax.contour(rlon[:],rlat[:],ow_lag, [0], 
            colors = 'red',linewidth = 5)


ax.set_ylim(0,30)
ax.set_yticks([0, 5, 10, 15, 20, 25, 30])
ax.set_yticklabels(y_tick_labels,fontsize = 15)
ax.set_xticks([95,105,115,125,135,145])
ax.set_xticklabels(x_tick_labels, fontsize = 15)
ax.tick_params(length=6)

ax.tick_params(bottom=True, top=True, left=True, right=True)
ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)

#ax.grid(linestyle = '--', linewidth = 0.2, color = 'black')
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(0.5)


plt.title("Date: {}".format(str(rv_date)) ,fontsize = 13,loc = 'left',y=1.04)
plt.title("Level: %s mb" %level,fontsize = 12, loc = 'right',y=1.04)

ax.add_feature(cfeature.COASTLINE.with_scale('50m'),
                            linewidths=0.5)
ax.add_feature(cfeature.BORDERS.with_scale('50m'),
                            linestyle='-',
                            linewidths=0.5)
 ax.add_feature(cfeature.LAND.with_scale('50m'),
                            color='gainsboro')

plt.savefig('snapshot_VortLag_20170716_00.eps')

#%%


#%%
plt.contourf(rlon[:],rlat[:],mask_frst)
# %%
# Phase 5: Create new startf file with this mask

# remove if old file exists
if os.path.exists(cdir + "startf_%s.4" %rv_date):
    os.remove(cdir + "startf_%s.4" %rv_date)

src_nc = netCDF4.Dataset(data_dir + "backward/startf_%s.4" %rv_date, mode = "r")
dst_nc = netCDF4.Dataset(cdir     + "startf_%s.4" %rv_date, mode = "w", format = "NETCDF4")
# %%
# length of ntra
lats,lons = np.where(mask_frst == 1)
lons = rlon[lons]
lats = rlat[lats]

# %%
# get source lon, lat ,p, level
lon_src = np.squeeze(src_nc.variables["lon"][:])
lat_src = np.squeeze(src_nc.variables["lat"][:])
lev_src = np.squeeze(src_nc.variables["level"][:])

arr = np.array([],dtype = int)
for ipt,rpts in enumerate(lats):
    new = np.where((lon_src == lons[ipt]) \
                  &(lat_src == lats[ipt])\
                  &(lev_src == level))
    arr = np.append(arr, new)
    # Flush
    new = None

#%%
# Copy global attributes all at once via dict
dst_nc.setncatts(src_nc.__dict__)

# copy dimensions except for dimx_lon
dst_nc.createDimension('dimx_lon',arr.shape[0])
for name, dimension in src_nc.dimensions.items():
    if name != 'dimx_lon':
        dst_nc.createDimension(name, len(dimension) if not dimension.isunlimited() else None)

for name, variable in src_nc.variables.items():
    dst_nc.createVariable(name, variable.datatype, variable.dimensions)
    
    if (name != 'time') & (name != 'BASEDATE'):
        for iarr,rarr in enumerate(arr):
            dst_nc[name][:,:,:,iarr] = src_nc[name][:,:,:,rarr]
    else:
        dst_nc[name][:] = src_nc[name][:]
        

    # copy variable attributes all at once via dictionary
    dst_nc[name].setncatts(src_nc[name].__dict__)

# %%
# close netCDF files
src_nc.close(); dst_nc.close()

# %%

# %%
