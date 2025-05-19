# coding=utf-8

# %%
# import libraries
import matplotlib.pyplot as plt
import numpy as np
from cartopy import feature as cfeature
from cartopy import crs as ccrs
from matplotlib.collections import LineCollection
from cartopy.mpl.geoaxes import GeoAxes, GeoAxesSubplot
from matplotlib.colors import BoundaryNorm
from matplotlib.pyplot import get_cmap

class CartoFigure:
    """ Wrapper to create maps based on cartopy
    """

    default_projection = ccrs.PlateCarree
    default_extent = [-180, 180, -90, 90]
    default_resolution = '10m'
    default_variable = 'p'

    def __init__(self, ax = None, projection = None, extent = None, resolution = None, **kwargs):
        """
        Parameters
        ----------
        ax: matplotlib.Axes
            If given use it to define the postion the GeoAx
        projection: cartopy.crs
            projection to override the defaut projection PlateCarree
        extent: list
            list defineing the domain extent;
            [minlon, maxlon, minlat, maxlat]
        resolution: string
            resolution to use for plotting the boundaries
            default 50m
        kwargs: Keyword arguments
            Keyword arguments to passs to plt.axes
        """
        self.projection = projection if projection else self.default_projection
        self.extent     = extent if extent else self.default_extent
        self.resolution = resolution if resolution else self.default_resolution

        if ax:
            self.ax = plt.axes(ax.get_position(), projection = self.position,
                            **kwargs)
            ax.set_visible(False)
        else:
            self.ax = plt.axes(projection=self.projection, **kwargs)
        if self.extent is not None:
            self.ax.set_extent(self.extent)

    def __getattr__(self,item):
        return getattr(self.ax, item)

    def __dir__(self):
        return self.ax.__dir__() + ['drawmap', 'plot_trajs']

    def drawmap(self):
        """ Draw the land feature
        Add from NaturalEarth the admin_0_countries
        """
        land = cfeature.NaturalEarthFeature('cultural', 'admin_0_countries',
                                    self.resoultion, edgecolor='gray',
                                    facecolor = 'none', linewidth = 0.5)
        self.ax.add_feature(land)

    def plot_trajs(self, trajs, variable = '', **kwargs):
        """ Plot trajectories on the map"""
        variable = variable if variable else self.default_variable
        kwargs['variable'] = variable
        return plot_trajs(self.ax, trajs, **kwargs)

def plot_trajs(ax, trajs, variable, cmap = 'Spectral', levels = None, **kwargs):
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

    Example
    -------
    from cartopy import crs as ccrs
    fig = plt.figure
    ax = plt.axes(projection = ccrs.PlateCarree(central_longitude = 45))
    lc = plot_trajs(ax, trajs = trajs, variable = 'z', cmap = 'viridis')
    """

    segments, colors = list(zip(*
                                [(_get_segments(traj),
                                traj[variable][:-1])
                                for traj in trajs]
                            ))
    # create a set of line segments so that we can color them individually
    # this creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments
    segments = np.concatenate(segments)
    colors   = np.concatenate(colors)
    cmap     = get_cmap(cmap)
    if levels is None:
        minlev = np.nanmin(trajs[variable])
        maxlev = np.nanmax(trajs[variable])
        levels = np.linspace(minlev,maxlev, 20)

    # Vreate a colormap and a norm to color
    norm = BoundaryNorm(levels, cmap.N) 

    nkwargs = {
            'array': colors,
            'cmap' : cmap,
            'norm' : norm
            }

    if isinstance(ax, GeoAxes):
        nkwargs['transform'] = ccrs.Geodetic()

    nkwargs.update(kwargs)
    lc = LineCollection(segments, **nkwargs)

    ax.add_collection(lc)
    #ax.add_colection(lc)

    return lc

def _get_segments(trajs):
    lon, lat = trajs['lon'], trajs['lat']
    points = np.array([lon,lat]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis = 1)
    return segments

            



# %%
