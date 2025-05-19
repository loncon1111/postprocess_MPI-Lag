import numpy as np
import xarray as xr

'''
second order accurate central differences in the interior points and 
first order (forward or backwards) differences at the boundaries
'''

class FTLE:
    """
    FTLE
    ====
    """

    def __init__(self, spherical_flag: bool, flag_3d: bool, integration_time_index: int,
                 dim_x: int, dim_y: int, dim_z: int):
        """
        args:
        - spherical_flag (bool): cartesian(false) spherical(true)
        - flag_3d (bool): 3d (True), 2d (false)
        - integration_time_index: integer number with the time index
        of the last position of the particles
        """
        self.spherical_flag = bool(spherical_flag)
        self.integration_time = integration_time_index
        self.flag_3d = bool(flag_3d)
        self.dim_x   = dim_x; self.dim_y = dim_y; self.dim_z = dim_z

    def get_integration_time(self, ds: xr.Dataset) -> [float, int]:
        """
        Get the integration in seconds from the integration time index
        
        args:
        - ds (xr.Dataset): dataset with lagrangian simulation
        returns:
        - T (float): integration time in seconds
        - timeindex (int): integration time index in time dataset axis
        """
        timeindex = self.integration_time
        print(timeindex)
        T = (ds.time.isel(time = timeindex) * 3600.).values.astype('f8')
        return T, timeindex

    def get_ftle_2d_cartesian(self, ds: xr.Dataset) -> np.array:
        """
        Get the 2D FTLE field in cartesian coordinates
        args:
        - ds (xr.Dataset): dataset with lagrangian simulation
        returns:
        - ftle (np.array): 2d dimensional array
        """
        T, timeindex = self.get_integration_time(ds)
        x_T = ds.x.isel(time = timeindex).values.squeeze()
        y_T = ds.y.isel(time = timeindex).values.squeeze()

        dx = np.gradient(ds.x0, axis = self.dim_x)
        dy = np.gradient(ds.y0, axis = self.dim_y) 

        dxdy = np.gradient(x_T, axis = self.dim_y)/dy
        dxdx = np.gradient(x_T, axis = self.dim_x)/dx

        dydy = np.gradient(y_T, axis = self.dim_y)/dy
        dydx = np.gradient(y_T, axis = self.dim_x)/dx

        ny = np.shape(dxdx)[self.dim_y], nx = np.shape(dxdx)[self.dim_x]
        ftle = np.zeros([ny,nx])
        for i in range(0, ny):
            for j in range(0, nx):
                # Jacobian-matrix
                J = np.array(
                    [[dxdx[i,j], dxdy[i,j]],
                     [dydx[i,j], dydy[i,j]]]
                )

                # Cauchy-Green strain tensor
                C = np.dot(np.transpose(J), J)
                eig_lya, _ = np.linalg.eigh(C)
                ftle[i,j] = (1./np.abs(T))*np.log(np.sqrt(eig_lya.max()))
        return ftle

    def get_ftle_2d_spherical(self, ds:xr.Dataset) -> np.array:
        """
        Get the 2D FTLE field in lat - lon coordinates.
        args:
        - ds (xr.Dataset): gridded dataset with lagrangian simulation
        returns:
        - ftle (np.array): 2d dimensional array
        """
        T, timeindex = self.get_integration_time(ds)
        R = 6.37e6
        x_T = ds.x.isel(time = timeindex).values.squeeze()
        y_T = ds.y.isel(time = timeindex).values.squeeze()

        dx = np.gradient(ds.x0, axis = self.dim_x)
        dy = np.gradient(ds.y0, axis = self.dim_y) 
        dxdy = np.gradient(x_T, axis = self.dim_y)/dy
        dxdx = np.gradient(x_T, axis = self.dim_x)/dx

        dydy = np.gradient(y_T, axis = self.dim_y)/dy
        dydx = np.gradient(y_T, axis = self.dim_x)/dx


        ny, nx = np.shape(dxdx)
        ftle = np.zeros([ny,nx])
        theta = ds.y.isel(time = timeindex).squeeze().values
        for i in range(0, ny):
            for j in range(0,nx):
                # Jacobian matrix
                J = np.array(
                    [[dxdx[i,j], dxdy[i,j]],
                    [dydx[i,j], dydy[i,j]]],
                    dtype = np.float32
                )
                #print(J)
                M = np.array(
                    [[R*R*np.cos(theta[i,j] * np.pi/180.), 0],
                    [0., R*R]],
                    dtype = np.float32
                )
                # Cauchy-Green strain tensor
                C = np.dot(np.dot(np.transpose(J),M), J)
                eig_lya, _ = np.linalg.eigh(C)
                ftle[i,j] = (1./np.abs(T)*np.log(np.sqrt(eig_lya.max())))
        return ftle

    def get_ftle_3d_cartesian(self, ds:xr.Dataset) -> np.array:
        """
        Gets the 3D FTLE field in cartesian coordinates.
        args:
        - ds (xr.Dataset): gridded dataset with lagrangian simulation
        returns:
        - ftle (np.array): 3d dimensional array
        """
        T, timeindex = self.get_integration_time(ds)
        x_T = ds.x.isel(time = timeindex).values.squeeze()
        y_T = ds.y.isel(time = timeindex).values.squeeze()
        z_T = ds.z.isel(time = timeindex).values.squeeze()

        dx = np.gradient(ds.x0, axis = self.dim_x)
        dy = np.gradient(ds.y0, axis = self.dim_y) 
        dz = np.gradient(ds.z0, axis = self.dim_z)

        dxdy = np.gradient(x_T, axis = self.dim_y)/dy
        dxdx = np.gradient(x_T, axis = self.dim_x)/dx
        dxdz = np.gradient(x_T, axis = self.dim_z)/dz

        dydy = np.gradient(y_T, axis = self.dim_y)/dy
        dydx = np.gradient(y_T, axis = self.dim_x)/dx
        dydz = np.gradient(y_T, axis = self.dim_z)/dz

        dzdy = np.gradient(z_T, axis = self.dim_y)/dy
        dzdx = np.gradient(z_T, axis = self.dim_x)/dx
        dzdz = np.gradient(z_T, axis = self.dim_z)/dz
        
        nz, ny, nx = dxdz.shape
        ftle = np.zeros([nz,ny,nx])

        for i in range(0,nz):
            for j in range(0,ny):
                for k in range(0,nx):
                    J = np.array(
                        [[dxdx[i,j,k], dxdy[i,j,k], dxdz[i,j,k]],
                         [dydx[i,j,k], dydy[i,j,k], dydz[i,j,k]],
                         [dzdx[i,j,k], dzdy[i,j,k], dzdz[i,j,k]]]
                    )
                    C = np.dot(np.transpose(J), J)
                    eig_lya, _ = np.linalg.eig(C)
                    ftle[i,j,k] = (1./np.abs(T)*np.log(np.sqrt(eig_lya.max())))

        return ftle

    def to_dataset(self, ds: xr.Dataset, ftle: np.array) -> xr.Dataset:
        """
        Write the FTLE field in xarray dataset
        args:
        - ds (xr.Dataset): gridded dataset with lagrangian simulation
        - ftle (np.array): 3d or 2d dimensional array
        returns:
        - ds (xr.Dataset): gridded dataset with ftle field
        """
        T, timeindex = self.get_integration_time
        if T > 0:
            ds['FTLE_forward'] = (ds.x.isel(time = timeindex).dims, ftle)
        if T < 0:
            ds['FTLE_backward'] = (ds.x.isel(time = timeindex).dims, ftle)
        return ds

    def get_ftle(self, ds: xr.Dataset, to_dataset = True):
        """
        It computes the FTLE (Finite time Lyapunov exponents) using the
        Cauchy Green finite time deformation tensor described

        args:
        - to_dataset (bool, optional): By default, it added the computed
        FTLE field, to the output dataset.
        returns:
        - ds (xr.Dataset): gridded dataset with ftle field
        """
        T, _ = self.get_integration_time(ds)
        t0 = ds.time.isel(time = 0).values

        print('-> FTLE >> Computing...')

        flag_3d = self.flag_3d

        if (self.spherical_flag is False) and (flag_3d is False):
            ftle = self.get_ftle_2d_cartesian(ds)
        elif (self.spherical_flag is True) and (flag_3d is False):
            ftle = self.get_ftle_2d_spherical(ds)
        elif (self.spherical_flag is False) and (flag_3d is True):
            ftle = self.get_ftle_3d_cartesian(ds)
        else:
            print('No spherical 3D FTLE available at the moment')
            print('Convert 3d spherical to cartesian please')
            return

        if to_dataset is True:
            self.to_dataset(ds, ftle)
            return ds
        else:
            return ftle

    def explore_ftle_timescale(self,ds, to_dataset = True):
        """
        It computes the FTLE for all timesteps instead of a given one.
        The output produced will help you to explore the timescale of the deformation
        in order to infer the attributes for LCS and FTLE computation.

        args:
        - to_dataset (bool,optional): by default, it added the computed FTLE field,
        to the output dataset.
        returns:
        self: ds_output with FTLE computed for all timesteps
        """

        ftle = np.zeros_like(ds.x.values)
        nsteps = ds.time.size

        for i in range(0, nsteps):
            self.integration_time = i
            ftle[i] = self.get_ftle(ds, to_dataset = False)
            print('->FTLE >> field for step ' + str(i))
        
        if to_dataset is True:
            ds['FTLE'] = (ds.x.dims, ftle)
            return ds
        else:
            return ftle

    def explore_ftle_2d_vertical(self, ds, to_dataset = True):
        ftle = np.zeros_like(ds.x.values)

        for i, z in enumerate(ds.zz0.values):
            ftle[i] = self.explore_ftle_timescale(ds.isel(zz0 = i), to_dataset = False)
            print('->FTLE >> field for level %s' %z)

        if to_dataset is True:
            ds['FTLE'] = (ds.x.dims, ftle)
            return ds
        else:
            return ftle
