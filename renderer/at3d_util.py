import pandas as pd
import numpy as np
import at3d
import xarray as xr
import functools

def load_from_csv(path, density=None, origin=(0.0,0.0)):

    df = pd.read_csv(path, comment='#', skiprows=4, index_col=['x', 'y', 'z'])
    nx, ny, nz = np.genfromtxt(path, skip_header=1, max_rows=1, dtype=int, delimiter=',')
    dx, dy = np.genfromtxt(path, max_rows=1, dtype=float, skip_header=2, delimiter=',')
    z = xr.DataArray(np.genfromtxt(path, max_rows=1, dtype=float, skip_header=3, delimiter=','), coords=[range(nz)], dims=['z'])

    dset = at3d.grid.make_grid(dx, nx, dy, ny, z)
    i, j, k = zip(*df.index)
    for name in df.columns:
        #initialize with np.nans so that empty data is np.nan
        variable_data = np.zeros((dset.sizes['x'], dset.sizes['y'], dset.sizes['z']))
        variable_data[i, j, k] = df[name]
        dset[name] = (['x', 'y', 'z'], variable_data)

    if density is not None:
        assert density in dset.data_vars, \
        "density variable: '{}' must be in the file".format(density)

        dset = dset.rename_vars({density: 'density'})
        dset.attrs['density_name'] = density

    dset.attrs['file_name'] = path

    return dset


def generate_at3d_mie_table(wavelength=0.672):
    mie_mono_table = at3d.mie.get_mono_table(
        'Water', (wavelength, wavelength),
        max_integration_radius=65.0,
        minimum_effective_radius=1,
        relative_dir='../AT3D/mie_tables',
        verbose=False
    )


    size_distribution_function = functools.update_wrapper(
        functools.partial(at3d.size_distribution.gamma,
                          normalization='geometric_extinction'),
        at3d.size_distribution.gamma
    )

    size_distribution = at3d.size_distribution.get_size_distribution_grid(
        mie_mono_table.radius,
        size_distribution_function=at3d.size_distribution.gamma,
        particle_density=1.0, radius_units='micron',
        reff=np.linspace(1.0, 100.0, 117),
        veff=np.linspace(0.01, 0.4, 117))


def hemispheric_projection(wavelength, resolution=5,
                           position_vector=np.array([0, 0, 0]),
                           stokes='I'):
    """
    Generates a sensor dataset that observes the hemisphere above.

    Parameters
    ----------
    wavelength: float,
        Wavelength in [micron]
    resolution: int
        This is not the Number of pixels in camera x and y axes.
        this is the Angular resolution of the measurements in [deg].
    position_vector: list of 3 float elements
        [x , y , z] which are:
        Location in global x coordinates [km] (North)
        Location in global y coordinates [km] (East)
        Location in global z coordinates [km] (Up)
    stokes: list or string
       list or string of stokes components to observe ['I', 'Q', 'U', 'V'].

    Returns
    -------
    sensor : xr.Dataset
        A dataset containing all of the information required to define a sensor
        for which synthetic measurements can be simulated;
        positions and angles of all pixels, sub-pixel rays and their associated weights,
        and the sensor's observables.

    """
    norm = lambda x: x / np.linalg.norm(x, axis=0)

    assert int(resolution) == resolution, "resolution is an integer >= 1"

    position = np.array(position_vector, dtype=np.float32)

    mu = np.cos(np.deg2rad(np.arange(0.0, 89.9, resolution)))
    phi = np.deg2rad(np.arange(0.0, 360.0, resolution))  # -resolution
    x, y, z, mu, phi = np.meshgrid(position[0], \
                                   position[1], \
                                   position[2], \
                                   mu, phi, indexing='ij')
    x = np.squeeze(x)
    y = np.squeeze(y)
    z = np.squeeze(z)
    phi = np.squeeze(phi)
    mu = np.squeeze(mu)

    nx = phi.shape[0]
    ny = phi.shape[1]

    x = x.ravel().astype(np.float32)
    y = y.ravel().astype(np.float32)
    z = z.ravel().astype(np.float32)
    mu = -mu.ravel().astype(np.float64)
    phi = phi.ravel().astype(np.float64)  # is it the same as (np.arctan2(y_c, x_c) + np.pi).astype(np.float64)?
    npix = x.size
    nrays = npix

    image_shape = [nx, ny]
    sensor = at3d.sensor.make_sensor_dataset(x.ravel(), y.ravel(), z.ravel(),
                                 mu.ravel(), phi.ravel(), stokes, wavelength)

    sensor['image_shape'] = xr.DataArray(image_shape,
                                         coords={'image_dims': ['nx', 'ny']},
                                         dims='image_dims')
    sensor.attrs = {
        'projection': 'Hemisphere',
        'fov_deg': None,
        'fov_x_deg': None,
        'fov_y_deg': None,
        'x_resolution': nx,
        'y_resolution': ny,
        'position': position,
        'lookat': None,
        'rotation_matrix': None,
        'projection_matrix': None,
        'sensor_to_camera_transform_matrix': None,
        'is_ideal_pointing': True  # CloudCT usage, default use is True, but if we add pointing noise, it is False.

    }

    pixel_index = range(len(x))
    ray_weight = np.ones_like(x)

    # update ray variables to sensor dataset.
    sensor['ray_mu'] = ('nrays', mu)
    sensor['ray_phi'] = ('nrays', phi)
    sensor['ray_x'] = ('nrays', x)
    sensor['ray_y'] = ('nrays', y)
    sensor['ray_z'] = ('nrays', z)
    sensor['pixel_index'] = ('nrays', pixel_index)
    sensor['ray_weight'] = ('nrays', ray_weight)
    sensor['use_subpixel_rays'] = True

    # Validation:
    # convert angles to rays directions:
    """
    import mayavi.mlab as mlab
    figh = mlab.figure()
    z_c = -mu
    x_c = -np.sin(np.arccos(mu))*np.cos(phi)
    y_c = -np.sin(np.arccos(mu))*np.sin(phi)

    mlab.quiver3d(x, y, z, x_c,y_c,z_c,\
                  line_width=2,color = (1.0, 0, 0), scale_factor=1,figure=figh)

    mlab.show()
    """
    # follow https://stackoverflow.com/questions/22561694/generating-a-hemispherical-surface-with-triangular-mesh-and-representing-a-data
    # to show the image on the hemisphere

    return sensor