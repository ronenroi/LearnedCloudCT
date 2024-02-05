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
        relative_dir='../at3d/mie_tables',
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
