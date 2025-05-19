#!/usr/bin/env python

# import libs
import numpy as np
import pandas as pd
import xarray as xr

class terr_hgt:
    def __init__(self, terrain_file: str, lat: np.array, lon: np.array, integration:int):
        self.terrain_file = terrain_file
        self.lat = lat; self.lon = lon
        self.ntime = len(lat)
        self.integration = integration
        if integration < 0:
            self.forward = False
        else:
            self.forward = True

    def _get_height(self):
        height = xr.open_dataset(self.terrain_file)["data"]
        traj_hgt = height.interp(lat = self.lat, lon = self.lon, method = 'linear').values.diagonal().squeeze()
        return traj_hgt

    def _offland_time(self):
        ntime = self.ntime; inte = self.integration
        actual_range = np.linspace(0,(ntime-1)*inte,ntime)
        traj_hgt = self._get_height()
        s = pd.Series((traj_hgt <= 0).astype(np.int32))
        #reverse s then take where it is zero where backward
        if self.forward is False:
            s = s.reindex(index = s.index[::-1])
            soffland = s.groupby(s.eq(0).cumsum()).cumsum()
            soffland = soffland.reindex(index = soffland.index[::-1]).to_numpy()
        else:
            soffland = s.groupby(s.eq(0).cumsum()).cumsum().to_numpy()

        offland = np.where(soffland == 1)
        time_offland = actual_range[offland]
        return offland, time_offland
 

