import at3d
import numpy as np
import xarray as xr
from collections import OrderedDict
import pylab as py
import pickle
import pandas as pd
import matplotlib.pyplot as plt
def load_from_csv_shdom(path, density=None, origin=(0.0,0.0)):

    df = pd.read_csv(path, comment='#', skiprows=3, delimiter=' ')
    nx, ny, nz = np.genfromtxt(path, skip_header=1, max_rows=1, dtype=int, delimiter=' ')
    dx, dy = np.genfromtxt(path, max_rows=1, usecols=(0, 1), dtype=float, skip_header=2)
    z_grid = np.genfromtxt(path, max_rows=1, usecols=range(2, 2 + nz), dtype=float, skip_header=2)
    z = xr.DataArray(z_grid, coords=[range(nz)], dims=['z'])

    dset = at3d.grid.make_grid(dx, nx, dy, ny, z)

    for index,name in zip([3,4,5],['lwc','reff','veff']):
        #initialize with np.nans so that empty data is np.nan
        variable_data = np.zeros((dset.sizes['x'], dset.sizes['y'], dset.sizes['z']))
        i=df.values[:,0].astype(int)
        j=df.values[:,1].astype(int)
        k=df.values[:,2].astype(int)

        variable_data[i, j, k] = df.values[:,index]
        dset[name] = (['x', 'y', 'z'], variable_data)

    if density is not None:
        assert density in dset.data_vars, \
        "density variable: '{}' must be in the file".format(density)

        dset = dset.rename_vars({density: 'density'})
        dset.attrs['density_name'] = density

    dset.attrs['file_name'] = path

    return dset

def load_from_csv(path, density=None, origin=(0.0,0.0)):

    df = pd.read_csv(path, comment='#', skiprows=4, index_col=['x', 'y', 'z'])
    nx, ny, nz = np.genfromtxt(path, skip_header=1, max_rows=1, dtype=int, delimiter=',')
    dx, dy = np.genfromtxt(path, max_rows=1, dtype=float, skip_header=2, delimiter=',')
    z = xr.DataArray(np.genfromtxt(path, max_rows=1, dtype=float, skip_header=3, delimiter=','), coords=[range(nz)], dims=['z'])

    dset = at3d.grid.make_grid(dx, nx, dy, ny, z)
    i, j, k = zip(*df.index)
    for name in df.columns:
        #initialize with np.nans so that empty data is np.nan
        variable_data = np.zeros((dset.sizes['x'], dset.sizes['y'], dset.sizes['z']))*np.nan
        variable_data[i, j, k] = df[name]
        dset[name] = (['x', 'y', 'z'], variable_data)

    if density is not None:
        assert density in dset.data_vars, \
        "density variable: '{}' must be in the file".format(density)

        dset = dset.rename_vars({density: 'density'})
        dset.attrs['density_name'] = density

    dset.attrs['file_name'] = path

    return dset


def get_projections(cfg=None):
    # if cfg.data.dataset_name == 'CASS_600CCN_roiprocess_10cameras_20m':
    #     path = '/wdata/roironen/Data/CASS_256x256x139_600CCN_50m_32x32x32_roipreprocess/10cameras_20m/shdom_projections2.pkl'
    # elif cfg.data.dataset_name == 'BOMEX_50CCN_10cameras_20m' or cfg.data.dataset_name == 'BOMEX_50CCN_aux_10cameras_20m' or cfg.data.dataset_name == 'BOMEX_10cameras_20m':
    #     path = '/wdata/roironen/Data/BOMEX_128x128x100_50CCN_50m_micro_256/10cameras_20m/shdom_projections.pkl'
    # elif cfg.data.dataset_name == 'HAWAII_2000CCN_10cameras_20m':
    #     path = '/wdata/roironen/Data/HAWAII_2000CCN_32x32x64_50m/10cameras_20m/shdom_projections.pkl'

    path = '/wdata/roironen/Data/BOMEX_128x128x100_50CCN_50m_micro_256/10cameras_20m/shdom_projections.pkl'
    # path = '/wdata/roironen/Data/HAWAII_2000CCN_32x32x64_50m/10cameras_20m/shdom_projections.pkl'

    with open(path, 'rb') as pickle_file:
        projection_list = pickle.load(pickle_file)['projections']
    sensor_dict = at3d.containers.SensorsDict()

    def perspective_projection(wavelength, projection, stokes='I'):
        """
        Generates a sensor dataset that observes a target location with
        a perspective (pinhole camera) projection.

        Parameters
        ----------
        wavelength: float,
            Wavelength in [micron]
        fov: float
            Field of view [deg]
        x_resolution: int
            Number of pixels in camera x axis
        y_resolution: int
            Number of pixels in camera y axis
        position_vector: list of 3 float elements
            [x , y , z] which are:
            Location in global x coordinates [km] (North)
            Location in global y coordinates [km] (East)
            Location in global z coordinates [km] (Up)
        lookat_vector: list of 3 float elements
            [x , y , z] which are:
            Point in global x coordinates [km] (North) where the camera is pointing at
            Point in global y coordinates [km] (East) where the camera is pointing at
            Point in global z coordinates [km] (Up) where the camera is pointing at
        up_vector: list of 3 float elements
            The up vector determines the roll of the camera.
        stokes: list or string
           list or string of stokes components to observe ['I', 'Q', 'U', 'V'].
        sub_pixel_ray_args : dict
            dictionary defining the method for generating sub-pixel rays. The callable
            which generates the position_perturbations and weights (e.g. at3d.sensor.gaussian)
            should be set as the 'method', while arguments to that callable, should be set as
            other entries in the dict. Each argument have two values, one for each of the
            x and y axes of the image plane, respectively.
            E.g. sub_pixel_ray_args={'method':at3d.sensor.gaussian, 'degree': (2, 3)}

        Returns
        -------
        sensor : xr.Dataset
            A dataset containing all of the information required to define a sensor
            for which synthetic measurements can be simulated;
            positions and angles of all pixels, sub-pixel rays and their associated weights,
            and the sensor's observables.

        """
        norm = lambda x: x / np.linalg.norm(x, axis=0)

        # assert samples>=1, "Sample per pixel is an integer >= 1"
        # assert int(samples) == samples, "Sample per pixel is an integer >= 1"
        nx = projection.resolution[0]
        ny = projection.resolution[1]
        assert int(nx) == nx, "x_resolution is an integer >= 1"
        assert int(ny) == ny, "y_resolution is an integer >= 1"

        # The bounding_box is not nessesary in the prespactive projection, but we still may consider
        # to use if we project the rays on the bounding_box when the differences in mu , phi angles are below certaine precision.
        #     if(bounding_box is not None):

        #         xmin, ymin, zmin = bounding_box.x.data.min(),bounding_box.y.data.min(),bounding_box.z.data.min()
        #         xmax, ymax, zmax = bounding_box.x.data.max(),bounding_box.y.data.max(),bounding_box.z.data.max()


        position = np.array(projection.position, dtype=np.float32)

        M = max(nx, ny)
        npix = nx * ny
        R = np.array([nx, ny]) / M  # R will be used to scale the sensor meshgrid.
        dy = 2 * R[1] / ny  # pixel length in y direction in the normalized image plane.
        dx = 2 * R[0] / nx  # pixel length in x direction in the normalized image plane.
        # x_s, y_s, z_s = np.meshgrid(np.linspace(-R[0] + dx / 2, R[0] - dx / 2, nx),
        #                             np.linspace(-R[1] + dy / 2, R[1] - dy / 2, ny), 1.0)
        x_s, y_s, z_s = np.meshgrid(np.linspace(-R[0], R[0] - dx, nx), np.linspace(-R[1], R[1] - dy, ny),
                                    1.0)
        # Here x_c, y_c, z_c coordinates on the image plane before transformation to the requaired observation angle
        fov = projection._fov
        focal = 1.0 / np.tan(
            np.deg2rad(fov) / 2.0)  # focal (normalized) length when the sensor size is 2 e.g. r in [-1,1).
        fov_x = np.rad2deg(2 * np.arctan(R[0] / focal))
        fov_y = np.rad2deg(2 * np.arctan(R[1] / focal))

        k = np.array([[focal, 0, 0],
                      [0, focal, 0],
                      [0, 0, 1]], dtype=np.float32)
        inv_k = np.linalg.inv(k)

        homogeneous_coordinates = np.stack([x_s.ravel(), y_s.ravel(), z_s.ravel()])

        x_c, y_c, z_c = norm(np.matmul(
            projection._rotation_matrix, np.matmul(inv_k, homogeneous_coordinates)))
        # Here x_c, y_c, z_c coordinates on the image plane after transformation to the requaired observation

        # x,y,z mu, phi in the global coordinates:
        mu = -z_c.astype(np.float64)
        phi = (np.arctan2(y_c, x_c) + np.pi).astype(np.float64)
        x = np.full(npix, position[0], dtype=np.float32)
        y = np.full(npix, position[1], dtype=np.float32)
        z = np.full(npix, position[2], dtype=np.float32)

        image_shape = [nx, ny]
        sensor = at3d.sensor.make_sensor_dataset(x.ravel(), y.ravel(), z.ravel(),
                                                 mu.ravel(), phi.ravel(), stokes, wavelength)
        # compare to orthographic projection, prespective projection may not have bounding box.
        #     if(bounding_box is not None):
        #         sensor['bounding_box'] = xr.DataArray(np.array([xmin,ymin,zmin,xmax,ymax,zmax]),
        #                                               coords={'bbox': ['xmin','ymin','zmin','xmax','ymax','zmax']},dims='bbox')

        sensor['image_shape'] = xr.DataArray(image_shape,
                                             coords={'image_dims': ['nx', 'ny']},
                                             dims='image_dims')
        sensor.attrs = {
            'projection': 'Perspective',
            'fov_deg': fov,
            'fov_x_deg': fov_x,
            'fov_y_deg': fov_y,
            'x_resolution': nx,
            'y_resolution': ny,
            'position': position,
            # 'lookat': lookat,
            'rotation_matrix': projection._rotation_matrix.ravel(),
            'sensor_to_camera_transform_matrix': k.ravel()

        }

        sensor = at3d.sensor._add_null_subpixel_rays(sensor)
        return sensor

    for projection in projection_list:
        sensor_dict.add_sensor('SideViewCamera',
                               perspective_projection(0.672, projection, stokes='I')
                               )

    return sensor_dict

### Make the RTE grid and medium microphysics.
# cloud_scatterer = load_from_csv('/wdata/roironen/Data/HAWAII_2000CCN_32x32x64_50m/10cameras_20m/cloud64785.txt',
#                                            density='lwc',origin=(0.0,0.0))
cloud_scatterer = load_from_csv('/wdata/roironen/Data/BOMEX_256x256x100_5000CCN_50m_micro_256/10cameras_20m/cloud0.txt',
                                           density='lwc',origin=(0.0,0.0))

#load atmosphere
atmosphere = xr.open_dataset('../../AT3D/data/ancillary/AFGL_summer_mid_lat.nc')
#subset the atmosphere, choose only the bottom four km.
reduced_atmosphere = atmosphere.sel({'z': atmosphere.coords['z'].data[atmosphere.coords['z'].data <= 5.0]})
#merge the atmosphere and cloud z coordinates
merged_z_coordinate = at3d.grid.combine_z_coordinates([reduced_atmosphere,cloud_scatterer])

# define the property grid - which is equivalent to the base RTE grid
rte_grid = at3d.grid.make_grid(cloud_scatterer.x.diff('x')[0],cloud_scatterer.x.data.size,
                          cloud_scatterer.y.diff('y')[0],cloud_scatterer.y.data.size,
                          merged_z_coordinate)


#finish defining microphysics because we can.

cloud_scatterer_on_rte_grid = at3d.grid.resample_onto_grid(rte_grid, cloud_scatterer)

# #We choose a gamma size distribution and therefore need to define a 'veff' variable.
size_distribution_function = at3d.size_distribution.gamma
#
# cloud_scatterer_on_rte_grid['veff'] = (cloud_scatterer_on_rte_grid.reff.dims,
#                                        np.full_like(cloud_scatterer_on_rte_grid.reff.data, fill_value=0.1))

#%% md
cloud_scatterer_on_rte_grid['veff'].values[cloud_scatterer_on_rte_grid['veff'].values >0.4] = 0.4
# cloud_scatterer_on_rte_grid['veff'].values[np.bitwise_and(cloud_scatterer_on_rte_grid['veff'].values < 0.1 , cloud_scatterer_on_rte_grid['veff'].values >0)] = 0.1
cloud_scatterer_on_rte_grid['reff'].values[cloud_scatterer_on_rte_grid['reff'].values <1] = 1
### Define Sensors
sensor_dict = get_projections()
wavelengths = sensor_dict.get_unique_solvers()


### get optical property generators


try:
    with open('/wdata/roironen/Data/at3d_mie.pkl', 'rb') as f:
        mie_mono_tables = pickle.load(f)
except:
    mie_mono_tables = OrderedDict()
    for wavelength in wavelengths:
        mie_mono_tables[wavelength] = at3d.mie.get_mono_table(
            'Water',(wavelength,wavelength),
            max_integration_radius=120.0,
            minimum_effective_radius=0.1,
            relative_dir='../../AT3D/mie_tables',
            verbose=True
        )
    with open('/wdata/roironen/Data/at3d_mie.pkl', 'wb') as f:
        pickle.dump(mie_mono_tables, f, protocol=pickle.HIGHEST_PROTOCOL)

optical_property_generator = at3d.medium.OpticalPropertyGenerator(
    'cloud',
    mie_mono_tables,
    size_distribution_function,
    reff=np.linspace(1.0,65.0,101),
    veff=np.linspace(0.01,0.4,101),
)
with open('/wdata/roironen/Data/at3d_optical_property_generator.pkl', 'wb') as f:
    pickle.dump(optical_property_generator, f, protocol=pickle.HIGHEST_PROTOCOL)
optical_properties = optical_property_generator(cloud_scatterer_on_rte_grid)



# one function to generate rayleigh scattering.
rayleigh_scattering = at3d.rayleigh.to_grid(wavelengths,reduced_atmosphere,rte_grid)



## Define Solvers - Define solvers last based on the sensor's spectral information.


# NB IF YOU REDEFINE THE SENSORS BUT KEEP THE SAME SET OF SOLVERS
# THERE IS NO NEED TO REDEFINE THE SOLVERS YOU CAN SIMPLY RERUN
# THE CELL BELOW WITHOUT NEEDING TO RERUN THE RTE SOLUTION.

solvers_dict = at3d.containers.SolversDict()
# note we could set solver dependent surfaces / sources / numerical_config here
# just as we have got solver dependent optical properties.
solar_azimuth= 45  # azimuth: 0 is beam going in positive X direction (North), 90 is positive Y (East).
sun_zenith= 150  # zenith: Solar beam zenith angle in range (90,180]
solarmu = np.cos(np.deg2rad(sun_zenith))
for wavelength in sensor_dict.get_unique_solvers():
    medium = {
        'cloud': optical_properties[wavelength],
        'rayleigh':rayleigh_scattering[wavelength]
     }
    config = at3d.configuration.get_config()
    solvers_dict.add_solver(
        wavelength,
        at3d.solver.RTE(
            numerical_params=config,
            surface=at3d.surface.lambertian(0.05),
            source=at3d.source.solar(wavelength,  solarmu, solar_azimuth),
            medium=medium,
            num_stokes=1#sensor_dict.get_minimum_stokes()[wavelength],
        )
   )


sensor_dict.get_measurements(solvers_dict, n_jobs=1, verbose=True)



### Visualize the observations


# The side view camera has some rays very close to horizontal.
# Because of the Cartesian geometry and flat surface these rays pass through a much
# longer optical path of rayleigh scatter than nearby rays creating a strong
# line of apparent radiance.


with open('/wdata/roironen/Data/BOMEX_256x256x100_5000CCN_50m_micro_256/10cameras_20m/train/cloud_results_0.pkl', 'rb') as f:
    pyshdom_images = pickle.load(f)['images']

for instrument in sensor_dict:
    sensor_images = sensor_dict.get_images(instrument)
    for sensor, shdom_im in zip(sensor_images,pyshdom_images):
        plt.imshow(sensor.I.values)
        plt.colorbar()
        py.show()
        plt.imshow(shdom_im)
        plt.colorbar()
        plt.show()



# Save the model. This saves the inputs required to form the solver object
# and also the sensor_data. The radiative transfer solutions themselves are not saved by default
# as they are large.
# See solver.save_solution / solver.load_solution() for how to save & load those RTE solutions.
# Those functions are not yet integrated into util.save_forward_model / util.load_forward_model.
at3d.util.save_forward_model('/wdata/roironen/Data/BOMEX_256x256x100_5000CCN_50m_micro_256/10cameras_20m/at3d.nc', sensor_dict, solvers_dict)
