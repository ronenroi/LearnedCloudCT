from shdom.generate import CloudGenerator
from shdom import float_round
import numpy as np
import shdom


class Monotonous(CloudGenerator):
    """
    Define a Clopud of Monotonous linear inclination in Reff and/or LWC.

    Parameters
    ----------
    args: arguments from argparse.ArgumentParser()
        Arguments required for this generator.
    """
    def __init__(self, args):
        super(Monotonous, self).__init__(args)

    @classmethod
    def update_parser(self, parser):
        """
        Update the argument parser with parameters relevant to this generator.

        Parameters
        ----------
        parser: argparse.ArgumentParser()
            The main parser to update.

        Returns
        -------
        parser: argparse.ArgumentParser()
            The updated parser.
        """
        parser.add_argument('--nx',
                            default=10,
                            type=int,
                            help='(default value: %(default)s) Number of grid cell in x (North) direction')
        parser.add_argument('--ny',
                            default=10,
                            type=int,
                            help='(default value: %(default)s) Number of grid cell in y (East) direction')
        parser.add_argument('--nz',
                            default=10,
                            type=int,
                            help='(default value: %(default)s) Number of grid cell in z (Up) direction')
        parser.add_argument('--domain_size',
                            default=1.0,
                            type=float,
                            help='(default value: %(default)s) Cubic domain size [km]')
        parser.add_argument('--extinction',
                            default=1.0,
                            type=np.float32,
                            help='(default value: %(default)s) Extinction [km^-1]')
        parser.add_argument('--CloudFieldFile',
                            default=None,
                            type=str,
                            help='(default value: %(default)s) Ground truth cloud')
        parser.add_argument('--lwc',
                            default=None,
                            type=np.float32,
                            help='(default value: %(default)s) Liquid water content of middle voxels [g/m^3]. '
                                 'If specified, extinction argument is ignored.')
        parser.add_argument('--reff',
                            default=8.0,
                            type=np.float32,
                            help='(default value: %(default)s) Effective radius of middle voxels[micron]')
        parser.add_argument('--min_reff',
                            default=3.0,
                            type=np.float32,
                            help='(default value: %(default)s) Min for reff[micron]')
        parser.add_argument('--min_lwc',
                            default=3.0,
                            type=np.float32,
                            help='(default value: %(default)s) Min for reff[micron]')
        parser.add_argument('--veff',
                            default=0.1,
                            type=np.float32,
                            help='(default value: %(default)s) Effective variance')
        return parser

    def get_cloudfield(self):
        '''
        Retrieve the groundtruth.

        Returns
        -------

        '''

    def get_grid(self):
        """
        Retrieve the scatterer grid.

        Returns
        -------
        grid: shdom.Grid
            A Grid object for this scatterer
        """
        bb = shdom.BoundingBox(0.0, 0.0, 0.0, self.args.domain_size, self.args.domain_size, self.args.domain_size)
        return shdom.Grid(bounding_box=bb, nx=self.args.nx, ny=self.args.ny, nz=self.args.nz)

    def get_extinction(self, wavelength=None, grid=None):
        """
        Retrieve the optical extinction at a single wavelength on a grid.

        Parameters
        ----------
        wavelength: float
            Wavelength in microns. A Mie table at this wavelength should be added prior (see add_mie method).
        grid: shdom.Grid, optional
            A shdom.Grid object. If None is specified than a grid is created from Arguments given to the generator (get_grid method)

        Returns
        -------
        extinction: shdom.GridData
            A GridData object containing the optical extinction on a grid

        Notes
        -----
        If the LWC is specified then the extinction is derived using (lwc,reff,veff). If not the extinction needs to be directly specified.
        The input wavelength is rounded to three decimals.
        """
        if grid is None:
            grid = self.get_grid()

        if self.args.lwc is None:
            if grid.type == 'Homogeneous':
                ext_data = self.args.extinction
            elif grid.type == '1D':
                ext_data = np.full(shape=(grid.nz), fill_value=self.args.extinction, dtype=np.float32)
            elif grid.type == '3D':
                ext_data = np.full(shape=(grid.nx, grid.ny, grid.nz), fill_value=self.args.extinction, dtype=np.float32)
            extinction = shdom.GridData(grid, ext_data)
        else:
            assert wavelength is not None, 'No wavelength provided'
            lwc = self.get_lwc(grid)
            reff = self.get_reff(grid)
            veff = self.get_veff(grid)
            extinction = self.mie[float_round(wavelength)].get_extinction(lwc, reff, veff)
        return extinction


    def get_lwc(self, grid=None, z0 = None):
        """
        Retrieve the liquid water content.
        Parameters
        ----------
        grid: shdom.Grid, optional
            A shdom.Grid object. If None is specified than a grid is created from Arguments given to the generator (get_grid method)

        z0 - scalar - cloud base height

        Returns
        -------
        lwc: shdom.GridData
            A GridData object containing liquid water content (g/m^3) on a 3D grid.
        """
        if grid is None:
            grid = self.get_grid()

        if z0 is None:
            z0 = grid.z[2] # when we do not pad droplets for the space curving step, the cloud base is likely to be at grid.z[2]
            # since when we cut the cloud from its field we pad 2 layers above and below the cloud.

        dz = (grid.zmax - grid.zmin )/ (grid.nz - 1)
        lwc = self.args.lwc
        if lwc is not None:
            if grid.type == '1D':
                Z = grid.z - (z0 - dz)
                Z[Z < 0] = 0
                lwc_data = (lwc * Z) + self.args.min_lwc

            elif grid.type == '3D':

                Z = grid.z - (z0 - dz)
                Z[Z < 0] = 0
                lwc_profile = (lwc * Z) + self.args.min_lwc
                lwc_data = np.tile(lwc_profile[np.newaxis, np.newaxis, :], (grid.nx, grid.ny, 1))

            lwc = shdom.GridData(grid, lwc_data)
        return lwc


    def get_reff(self, grid=None, z0 = None):
        """
        Retrieve the effective radius on a grid.
        Parameters
        ----------
        grid: shdom.Grid, optional
            A shdom.Grid object. If None is specified than a grid is created from Arguments given to the generator (get_grid method)

        z0 - scalar - cloud base height

        Returns
        -------
        reff: shdom.GridData
            A GridData object containing the effective radius [microns] on a grid
        """
        reff = self.args.reff
        if grid is None:
            grid = self.get_grid()

        if z0 is None:
            z0 = grid.z[2] # when we do not pad droplets for the space curving step, the cloud base is likely to be at grid.z[2]
            # since when we cut the cloud from its field we pad 2 layers above and below the cloud.

        dz = (grid.zmax - grid.zmin )/ (grid.nz - 1)
        if grid.type == '1D':
            Z = grid.z - (z0 - dz)
            Z[Z < 0] = 0
            reff_data = (reff * Z ** (1. / 3.)) + self.args.min_reff
            # reff_data = np.squeeze(np.array([np.linspace(reff-dreff, reff+dreff, num=grid.nz)]))
        elif grid.type == '3D':
            Z = grid.z - (z0 - dz)
            Z[Z < 0] = 0
            reff_profile = (reff * Z ** (1. / 3.)) + self.args.min_reff
            # reff_profile = np.array([np.linspace(reff-dreff, reff+dreff, num=grid.nz)])
            reff_data = np.tile(reff_profile[np.newaxis, np.newaxis, :], (grid.nx, grid.ny, 1))
        return shdom.GridData(grid, reff_data)

    def get_veff(self, grid=None):
        """
        Retrieve the effective variance on a grid.

        Parameters
        ----------
        grid: shdom.Grid, optional
            A shdom.Grid object. If None is specified than a grid is created from Arguments given to the generator (get_grid method)

        Returns
        -------
        veff: shdom.GridData
            A GridData object containing the effective variance on a grid
        """
        if grid is None:
            grid = self.get_grid()
        if grid.type == 'Homogeneous':
            veff_data = self.args.veff
        elif grid.type == '1D':
            veff_data = np.full(shape=(grid.nz), fill_value=self.args.veff, dtype=np.float32)
        elif grid.type == '3D':
            veff_data = np.full(shape=(grid.nx, grid.ny, grid.nz), fill_value=self.args.veff, dtype=np.float32)

        return shdom.GridData(grid, veff_data)

    def get_phase(self, wavelength, mask, grid=None):
        """
        Retrieve the phase function at a single wavelength on a grid.

        Parameters
        ----------
        wavelength: float
            Wavelength in microns. A Mie table at this wavelength should be added prior (see add_mie method).
        grid: shdom.Grid, optional
            A shdom.Grid object. If None is specified than a grid is created from Arguments given to the generator (get_grid method)

        Returns
        -------
        phase: shdom.GridPhase
            A GridPhase object containing the phase function on a grid

        Notes
        -----
        The input wavelength is rounded to three decimals.
        """
        if grid is None:
            grid = self.get_grid()
        reff = self.get_reff(grid)
        reff._data[np.bitwise_not(mask)] = 1
        veff = self.get_veff(grid)
        veff._data[np.bitwise_not(mask)] = 0.01

        return self.mie[float_round(wavelength)].get_phase(reff, veff)