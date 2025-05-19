#coding=utf-8
# %% import libraries
from metpy.calc.thermo import thickness_hydrostatic_from_relative_humidity
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
import metpy.calc as mpcalc
import metpy.units as units
import netCDF4
import xarray as xr
import datetime
import glob
import function_traject


# functions for dates
def next_time(startdate,days=0,hours=0,minutes=0,seconds=0):
    """
    Find next time for calculation
    """
    sdate = datetime.strptime(startdate, '%Y%m%d_%H')
    date  = sdate + datetime.timedeltatimedelta(days=days, hours=hours)
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

# functions for plotting
def panel_plot(ax, variable, cmap = 'Spectral', levels = None,
               xrange = None,yrange = None,mask = None, **kwargs):
    """ Plot trajectories on axis
    Parameters
    ----------
    ax:
    trajs
    variable: string
    cmap: string
    levels: ndarray
    transform: CRS (Coordinate Reference System) object, default ccrs.Geodetic()
    kwargs: dict,
        passed to LineCollection
    """
    import cartopy.feature as cfeature
    import cartopy.crs as ccrs

    cmap     = plt.get_cmap(cmap)

    if levels is None:
        minlev = np.nanmin(variable)
        maxlev = np.nanmax(variable)
        levels = np.linspace(minlev,maxlev, 21)

    try:
        lons = variable.lon.values
        lats = variable.lat.values
        lev  = variable.lev.values
    except AttributeError:
        lons = variable.longitude.values
        lats = variable.latitude.values
        lev  = variable.level.values

    if xrange is None:
        xrange = np.linspace(lons[0],lons[-1],20)
    if yrange is None:
        yrange = np.linspace(lats[0],lats[-1],20)

    # lon,lat tick labels
    x_tick_labels = [u'%s\N{DEGREE SIGN}' %x for x in xrange]
    y_tick_labels = [u'%s\N{DEGREE SIGN}' %y for y in yrange]



    # plot contour
    cs = ax.contourf(lons,lats,variable,
                     levels = levels,
                     cmap = cmap,
                     extend = 'both'
                     )
    cs.monochrome = True

    # plot masking contour if available
    if mask is not None:
        cf = ax.contour(lons,lats,mask,
                        levels = [0,1],
                        colors = 'black')

    ax.set_ylim(yrange[0],yrange[-1])
    ax.set_xlim(xrange[0],xrange[-1])
    ax.set_yticks(yrange)
    ax.set_yticklabels(y_tick_labels,fontsize = 18)
    ax.set_xticks(xrange)
    ax.set_xticklabels(x_tick_labels,fontsize = 18)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)

    # add geographical features
    ax.add_feature(cfeature.BORDERS.with_scale('10m'),
                   linestyle = '-',
                   linewidths = 0.5)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'),
                   linestyle = '-',
                   linewidths = 0.8)

    anchored_text = AnchoredText("%i hPa" %lev, 
                                loc = 2,
                                prop = dict(fontweight="bold",fontsize=18),
                                frameon = True)
    ax.add_artist(anchored_text)
    return cs

def _traj_array(ncname):
    ncfile = netCDF4.Dataset(ncname)
    ntra, ntime = function_traject._get_netcdf_traj_dim(ncfile)

    exclude = ['BASEDATE','time']
    variables = [var for var in ncfile.variables if var not in exclude]
    formats   = [ncfile.variables[var].dtype for var in variables]

    traj_levels = np.squeeze(ncfile.variables['p'])
    traj_array = np.zeros((ntra,ntime), dtype = {'names'   : variables,
                                                 'formats' : formats})
    for var in variables:
        vardata = ncfile.variables[var]
        msv = vardata.missing_data
        vardata = np.squeeze(vardata).T 
        #vardata = np.squeeze(vardata).T
        vardata[vardata <= msv] = np.nan
        traj_array[var] = vardata
    return traj_array

def _init_mask(variable):
    
    try:
        lons  = variable.longitude
        lats  = variable.latitude
        level = variable.level
    except AttributeError:
        lons  = variable.lon
        lats  = variable.lat
        level = variable.lev

    mask = np.empty(variable.shape,np.int)
    mask_array = xr.DataArray(
                data = mask,
                coords = variable.coords,
                dims   = variable.dims,
                attrs  = dict(
                        standard_name = 'masking_array_vortices',
                        long_name     = 'Masking Array for Vortices'
                )
    )
    return mask_array

def _get_mask(ncname,mask_array,labels = None):
    # define the coordinates and level
    try:
        rlon = mask_array.longitude.values
        rlat = mask_array.latitude.values
        rlev = mask_array.level.values
    except AttributeError:
        rlon = mask_array.lon.values
        rlat = mask_array.lat.values
        rlev = mask_array.lev.values

    traj_array = _traj_array(ncname)
    new_mask = mask_array.values

    ntra,ntime = traj_array.shape

    if labels is None:
        labels = np.ones(ntra)

    for ilev,lev in enumerate(rlev):
        index_sel = np.where(traj_array['p'][:,0] == lev)

        filt_traj = traj_array[index_sel,:]
    
        frst_lon = np.squeeze(traj_array['lon'][index_sel,0])
        frst_lat = np.squeeze(traj_array['lat'][index_sel,0])

        lonind = [np.where(rlon == x)[0][0] for x in frst_lon]
        latind = [np.where(rlat == x)[0][0] for x in frst_lat]
    
        if len(lonind) > 0:
            new_mask[ilev,latind,lonind] = labels[index_sel]

        # Flush
        index_sel, filt_traj = 2*[None]; frst_lon,frst_lat,lonind,latind = 4*[None]

    # redefine xarray new_mask
    new_mask = xr.DataArray(
                data = new_mask,
                coords = mask_array.coords,
                dims   = mask_array.dims,
                attrs  = dict(
                        standard_name = 'masking_array_vortices',
                        long_name     = 'Masking Array for Vortices'
                )
    )
    return new_mask


def _get_netcdf_traj_dim(filename):
    """ return number of trajectories (ntraj) and number of time step (ntime)"""
    ncfile = netCDF4.Dataset(filename)
    dim_set = {'dimx_lon', 'id', 'ntra'}
    dim_nc  = set(ncfile.dimensions.keys())

    try:
        ntra_dim = dim_set.intersection(dim_nc).pop()
        ntra = len(ncfile.dimensions[ntra_dim])
    except KeyError:
        raise Exception('Cannot read the number of trajectories, ' +
                        'not one of (' + ' '.join(dim_set) + ')')
            
    try: 
        ntime = len(ncfile.dimensions['time'])
    except KeyError:
        ntime = len(ncfile.dimensions['ntim'])

    return ntra, ntime

def _cal_the(data,lon_slice,lat_slice,all_levs):
    q = np.squeeze(data.Q.metpy.sel(
                    latitude = lat_slice,
                    longitude = lon_slice,
                    level = all_levs)[:]
                    )
    t = np.squeeze(data.T.metpy.sel(
                    latitude = lat_slice,
                    longitude = lon_slice,
                    level = all_levs)[:]
                    )
    msl = data["msl"].sel(
                    latitude = lat_slice,
                    longitude = lon_slice,
                    )

    levs,lats,lons = t.indexes.values()
    nlon = np.shape(t)[2]
    nlat = np.shape(t)[1]
    p = np.tile(all_levs[:,np.newaxis,np.newaxis], (1,nlat,nlon))
    p = xr.DataArray(
        data   = p, 
        coords = t.coords, 
        dims   = t.dims, 
        attrs  = dict(
                standard_name = 'Pressure',
                long_name     = 'pressure at isobaric level',
                units         = 'hPa'
            ),
        )
    tdew = mpcalc.dewpoint_from_specific_humidity(q,t,p)
    the  = mpcalc.equivalent_potential_temperature(p,t,tdew)
    #equiv_th = the.magnitude
    the = xr.DataArray(
        data   = the.magnitude,
        coords = t.coords,
        dims   = t.dims,
        attrs  = dict(
            standard_name = 'equiv_th',
            long_name     = 'Equivalent Potential Temperature',
            units         = t.units
        ) 
    )
    return the
    
# %%

# %%
