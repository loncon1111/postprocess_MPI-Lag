# %%
import netCDF4
import numpy as np
import math
# %%
cdir = "/Users/daohoa/Desktop/my-notebook/LAGRANGIAN/TSCOMBO2017/vortex_regions/"
rv_date = "20170714_18"
# %%

lsl_file = netCDF4.Dataset(cdir + "lsl_%s_clt.4" %(rv_date), mode = 'r+')
# %%
vars = []
ref_dims = ('time','dimz_lon','dimy_lon','dimx_lon')
for ivar,var in enumerate(lsl_file.variables):
    print(var)
    if lsl_file.variables[var].dimensions == ref_dims:
        vars.append(var)
# %%
# Get needed variables
lat  = lsl_file.variables['lat'][:]
u    = lsl_file.variables['U'][:]
dudx = lsl_file.variables['DUDX'][:]
dudy = lsl_file.variables['DUDY'][:]
dvdx = lsl_file.variables['DVDX'][:]
dvdy = lsl_file.variables['DVDY'][:]
dudp = lsl_file.variables['DUDP'][:]
dvdp = lsl_file.variables['DVDP'][:]
dthdp = lsl_file.variables['DTHDP'][:]
dthdx = lsl_file.variables['DTHDX'][:]
dthdy = lsl_file.variables['DTHDY'][:]
th   = lsl_file.variables['TH'][:]
the  = lsl_file.variables['THE'][:]
# %%
# Calculate absvort
deltay = 1.112E5
omega = 7.292E-5
pi180 = math.pi/180
absvort = dvdx - dudy + u*pi180/deltay*np.tan(pi180 * lat) + \
        2 * omega * np.sin(lat*pi180)
print(absvort)
# %%
# Calculate potential vorticity PV
scale = 1.E6 #(PVU)
g = 9.80655
pvort = - scale * g * (absvort * dthdp + dudp * dthdy - dvdp * dthdx)
print(pvort)
# %%
absv = lsl_file.createVariable('ABSVORT', absvort.dtype, ref_dims)
absv.units = 's**-1'
absv.long_name = 'Absolute Vorticity'

# %%
pvor = lsl_file.createVariable('PV', pvort.dtype, ref_dims)
pvor.units = 'PVU'
pvor.long_name = 'Potential Vorticity'

# %%
# copy attributes
absv.setncatts(lsl_file['U'].__dict__)
pvor.setncatts(lsl_file['U'].__dict__)
# %%
lsl_file.variables
# %%
# %%
pvor[:] = pvort
absv[:] = absvort
# %%
lsl_file.close()
# %%
