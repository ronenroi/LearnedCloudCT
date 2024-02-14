import at3d
import mayavi.mlab as mlab
from collections import OrderedDict
from shdom.CloudCT_Utils import *


def save_to_csv(cloud_scatterer, file_name, comment_line='', OLDPYSHDOM=False):
    """

    A utility function to save a microphysical medium.
    After implementation put as a function in util.py under the name
    save_to_csv.

    Format:


    Parameters
    ----------
    path: str
        Path to file.
    comment_line: str, optional
        A comment line describing the file.
    OLDPYSHDOM: boll, if it is True, save the txt in old version of pyshdom.

    Notes
    -----
    CSV format is as follows:
    # comment line (description)
    nx,ny,nz # nx,ny,nz
    dx,dy # dx,dy [km, km]
    z_levels[0]     z_levels[1] ...  z_levels[nz-1]
    x,y,z,lwc,reff,veff
    ix,iy,iz,lwc[ix, iy, iz],reff[ix, iy, iz],veff[ix, iy, iz]
    .
    .
    .
    ix,iy,iz,lwc[ix, iy, iz],reff[ix, iy, iz],veff[ix, iy, iz]



    """
    xgrid = cloud_scatterer.x
    ygrid = cloud_scatterer.y
    zgrid = cloud_scatterer.z

    dx = cloud_scatterer.delx.item()
    dy = cloud_scatterer.dely.item()
    dz = round(np.diff(zgrid)[0], 5)

    REGULAR_LWC_DATA = np.nan_to_num(cloud_scatterer.lwc)
    REGULAR_REFF_DATA = np.nan_to_num(cloud_scatterer.reff)
    REGULAR_VEFF_DATA = np.nan_to_num(cloud_scatterer.veff)

    y, x, z = np.meshgrid(range(cloud_scatterer.sizes.get('y')), \
                          range(cloud_scatterer.sizes.get('x')), \
                          range(cloud_scatterer.sizes.get('z')))

    if not OLDPYSHDOM:

        with open(file_name, 'w') as f:
            f.write(comment_line + "\n")
            # nx,ny,nz # nx,ny,nz
            f.write('{}, {}, {} '.format(int(cloud_scatterer.sizes.get('x')), \
                                         int(cloud_scatterer.sizes.get('y')), \
                                         int(cloud_scatterer.sizes.get('z')), \
                                         ) + "# nx,ny,nz\n")
            # dx,dy # dx,dy [km, km]
            f.write('{:2.3f}, {:2.3f} '.format(dx, dy) + "# dx,dy [km, km]\n")

            # z_levels[0]     z_levels[1] ...  z_levels[nz-1]

            np.savetxt(f, \
                       X=np.array(zgrid).reshape(1, -1), \
                       fmt='%2.3f', delimiter=', ', newline='')
            f.write(" # altitude levels [km]\n")
            f.write("x,y,z,lwc,reff,veff\n")

            data = np.vstack((x.ravel(), y.ravel(), z.ravel(), \
                              REGULAR_LWC_DATA.ravel(), REGULAR_REFF_DATA.ravel(), REGULAR_VEFF_DATA.ravel())).T
            # Delete unnecessary rows e.g. zeros in lwc
            mask = REGULAR_LWC_DATA.ravel() > 0
            data = data[mask, ...]
            np.savetxt(f, X=data, fmt='%d ,%d ,%d ,%.5f ,%.3f ,%.5f')

    else:
        # save in the old version:
        with open(file_name, 'w') as f:
            f.write(comment_line + "\n")
            # nx,ny,nz # nx,ny,nz
            f.write('{} {} {} '.format(int(cloud_scatterer.sizes.get('x')), \
                                       int(cloud_scatterer.sizes.get('y')), \
                                       int(cloud_scatterer.sizes.get('z')), \
                                       ) + "\n")

            # dx,dy ,z
            np.savetxt(f, X=np.concatenate((np.array([dx, dy]), zgrid)).reshape(1, -1), fmt='%2.3f')
            # z_levels[0]     z_levels[1] ...  z_levels[nz-1]

            data = np.vstack((x.ravel(), y.ravel(), z.ravel(), \
                              REGULAR_LWC_DATA.ravel(), REGULAR_REFF_DATA.ravel(), REGULAR_VEFF_DATA.ravel())).T
            # Delete unnecessary rows e.g. zeros in lwc
            mask = REGULAR_LWC_DATA.ravel() > 0
            data = data[mask, ...]
            np.savetxt(f, X=data, fmt='%d %d %d %.5f %.3f %.5f')

# -----------------------------------------------------
# -------- Input parameters by the user:---------------
# -----------------------------------------------------
IFVISUALIZE = False

dx = 0.05 #km
dy = 0.05 #km
dz = 0.04 #km
# -----------------------------------------------------
# -----------------------------------------------------
# -----------------------------------------------------
# -----------------------------------------------------
# -----------------------------------------------------
wavelength_band = (0.66, 0.66)
wavelen1, wavelen2 = wavelength_band
wavelength_averaging = False

formatstr = 'TEST_Water_{}nm.nc'.format(int(1e3*wavelength_band[0]))
if not os.path.exists('../AT3D/mie_tables'):
    os.makedirs('../AT3D/mie_tables')
if not (wavelen1 == wavelen2):
    wavelength_averaging = True   
    formatstr = 'TEST_averaged_Water_{}-{}nm.nc'.format(int(1e3*wavelength_band[0]), int(1e3*wavelength_band[1]))
mono_path = os.path.join('../AT3D/mie_tables', formatstr)


file_path = "../data/........"
data = sio.loadmat(file_path)
"""
There are 5 clouds:
'beta_est', 'beta_m_1std', 'beta_m_2std', 'beta_p_1std', 'beta_p_2std'])

"""

beta1 = data['beta_est']
beta2 = data['beta_m_1std']
beta3 = data['beta_m_2std']
beta4 = data['beta_p_1std']
beta5 = data['beta_p_2std']

#beta_sample = beta1
#file_name = 'beta_est.txt'
ref_folder = '../data/........'

for file_name_stamp in ['beta_est','beta_m_1std','beta_m_2std','beta_p_1std','beta_p_2std']:
    beta_sample= data[file_name_stamp]
    file_name = file_name_stamp+'.txt' # to save for the old version, use 'old_pyshdom_'+
    file_name = os.path.join(ref_folder, file_name)

    #-------------------------------------
    #-------------------------------------
    #-------------------------------------
    #-------------------------------------
    
    nx, ny, nz = beta_sample.shape
    
    z = np.linspace(0, (nz-1)*dz, nz)
    
    #Convert medium to pyshdom:
    # set grid using new pyshdom:
    # make a grid for microphysics which is just the cloud grid.
    cloud_scatterer = at3d.grid.make_grid(dx,nx,\
                              dy,ny,z)
    
    
    reff_data = np.full(shape=(nx, ny, nz), fill_value=10, dtype=np.float32)
    lwc_data = np.full(shape=(nx, ny, nz), fill_value=1, dtype=np.float32)
    veff_data = np.full(shape=(nx, ny, nz), fill_value=0.1, dtype=np.float32)
    
    DATA_DICT = OrderedDict()
    DATA_DICT['density']  = lwc_data # this is a unit lwc
    DATA_DICT['reff'] = reff_data
    DATA_DICT['veff'] = veff_data
    
    non_zero_indexes = np.where(beta_sample>0)
    i, j, k = non_zero_indexes
    
    for data_name in ('density' , 'reff', 'veff'):
        #initialize with np.nans so that empty data is np.nan    
        this_data = np.zeros((cloud_scatterer.sizes['x'], \
                    cloud_scatterer.sizes['y'], cloud_scatterer.sizes['z']))*np.nan
        this_data[i, j, k] = DATA_DICT[data_name][i, j, k]
        cloud_scatterer[data_name] = (['x', 'y', 'z'], this_data)
        
    
    xgrid = cloud_scatterer.x
    ygrid = cloud_scatterer.y
    zgrid = cloud_scatterer.z
        
    # We choose a gamma size distribution and therefore need to define a 'veff' variable.
    # make a grid for microphysics which is just the cloud grid.
    rte_grid = at3d.grid.make_grid(dx,cloud_scatterer.x.data.size,
                                   dy,cloud_scatterer.y.data.size,
                              cloud_scatterer.z)
    
    cloud_scatterer_on_rte_grid = at3d.grid.resample_onto_grid(rte_grid, cloud_scatterer)
    
    
    size_distribution_function = at3d.size_distribution.gamma
    
    # Exact OpticalPropertyGenerator:
    # get_mono_table will first search a directory to see if the requested table exists otherwise it will calculate it. 
    # You can save it to see if it works.
    mie_mono_table = at3d.mie.get_mono_table(
        'Water',wavelength_band,
        max_integration_radius=65.0,
        wavelength_averaging = wavelength_averaging, 
        minimum_effective_radius=0.1,
        relative_dir='../mie_tables',
        verbose=False
    )
    
    mie_mono_table.to_netcdf(mono_path)
    mie_mono_tables = OrderedDict()
    mie_mono_tables[wavelength_band[0]] = mie_mono_table
      
    optical_prop_gen = at3d.medium.OpticalPropertyGenerator(
        'cloud',
        mie_mono_tables, 
        size_distribution_function,
        particle_density=1.0, 
        maxnphase=None,
        interpolation_mode='exact',
        density_normalization='density',#The density_normalization argument is a convenient
        reff=np.linspace(1,20.0,20),
        veff=np.linspace(0.05,0.15,15)
    )
    
    optical_properties = optical_prop_gen(cloud_scatterer_on_rte_grid)
    # The optical properties produced by this contain all of the information 
    # required for the RTE solver. Note that the attributes track the inputs
    # and the microphysical properties are also brought along for traceability
    # purposes.
    
    # If you generate your own optical properties they must pass this check to be used in the solver.
    at3d.checks.check_optical_properties(optical_properties[wavelength_band[0]])
    
    # Extinction:
    extinction = np.array(optical_properties[wavelength_band[0]].extinction)
    
    
    #----------------------------------
    #----------------------------------
    #----------------------------------
    
    
    real_lwc = beta_sample/extinction
    this_data = cloud_scatterer['density'].values * real_lwc
    cloud_scatterer['density'] = (['x', 'y', 'z'], this_data)
    
    
    
    
    
    # check if i set the extinction correctly:
    rte_grid = at3d.grid.make_grid(dx,cloud_scatterer.x.data.size,
                                   dy,cloud_scatterer.y.data.size,
                              cloud_scatterer.z)
    
    cloud_scatterer_on_rte_grid = at3d.grid.resample_onto_grid(rte_grid, cloud_scatterer)
    optical_properties = optical_prop_gen(cloud_scatterer_on_rte_grid)
    
    # If you generate your own optical properties they must pass this check to be used in the solver.
    at3d.checks.check_optical_properties(optical_properties[wavelength_band[0]])
    
    # Extinction:
    test_extinction = np.array(optical_properties[wavelength_band[0]].extinction)
    
    
    #--------------------------------------------------
    
    
    if IFVISUALIZE:
        REGULAR_LWC_DATA = np.nan_to_num(test_extinction)
        
        mlab.figure(size=(600, 600))
        X, Y, Z = np.meshgrid(xgrid, ygrid, zgrid, indexing='ij')
        figh = mlab.gcf()
        src = mlab.pipeline.scalar_field(X, Y, Z, REGULAR_LWC_DATA, figure=figh)
        src.spacing = [dx, dy, dz]
        src.update_image_data = True
        
        isosurface = mlab.pipeline.iso_surface(src, contours=[0.1*REGULAR_LWC_DATA.max(),\
                                                              0.2*REGULAR_LWC_DATA.max(),\
                                                              0.3*REGULAR_LWC_DATA.max(),\
                                                              0.4*REGULAR_LWC_DATA.max(),\
                                                              0.5*REGULAR_LWC_DATA.max(),\
                                                              0.6*REGULAR_LWC_DATA.max(),\
                                                              0.7*REGULAR_LWC_DATA.max(),\
                                                              0.8*REGULAR_LWC_DATA.max(),\
                                                              0.9*REGULAR_LWC_DATA.max(),\
                                                              ],opacity=0.9,figure=figh)
        
        mlab.colorbar()
        
        
        
        REGULAR_LWC_DATA = np.nan_to_num(beta_sample)    
        mlab.figure(size=(600, 600))
        X, Y, Z = np.meshgrid(xgrid, ygrid, zgrid, indexing='ij')
        figh = mlab.gcf()
        src = mlab.pipeline.scalar_field(X, Y, Z, REGULAR_LWC_DATA, figure=figh)
        src.spacing = [dx, dy, dz]
        src.update_image_data = True
        
        isosurface = mlab.pipeline.iso_surface(src, contours=[0.1*REGULAR_LWC_DATA.max(),\
                                                              0.2*REGULAR_LWC_DATA.max(),\
                                                              0.3*REGULAR_LWC_DATA.max(),\
                                                              0.4*REGULAR_LWC_DATA.max(),\
                                                              0.5*REGULAR_LWC_DATA.max(),\
                                                              0.6*REGULAR_LWC_DATA.max(),\
                                                              0.7*REGULAR_LWC_DATA.max(),\
                                                              0.8*REGULAR_LWC_DATA.max(),\
                                                              0.9*REGULAR_LWC_DATA.max(),\
                                                              ],opacity=0.9,figure=figh)
        
        mlab.colorbar()    
        mlab.show()
        
        
    # save the files:
    comment_line = '# To measure flux divietions for ICCP rebutle Original cloud is unknown'
    # rename density to lwc:
    cloud_scatterer = cloud_scatterer.rename_vars({'density':'lwc'})
    save_to_csv(cloud_scatterer, file_name, comment_line)
    #save_to_csv(cloud_scatterer, file_name, comment_line, OLDPYSHDOM = True)

print("done")