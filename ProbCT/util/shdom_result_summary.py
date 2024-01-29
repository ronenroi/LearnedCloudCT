import pickle
import matplotlib.pyplot as plt
import glob, os
import scipy.io as sio


import numpy as np






if __name__ == "__main__":
    if False:
        path = '/wdata/roironen/Data/SHDOM/BOMEX_256x256x100_5000CCN_50m_micro_256/10cameras_50m'
        eps = [1]
    else:
        path = '/wdata/roironen/Data/SHDOM/CASS_50m_256x256x139_600CCN/10cameras_50m'
        eps = np.load('/wdata/roironen/Data/SHDOM/clouds.npy').tolist()
    result_path = glob.glob(os.path.join(path, "*.mat"))
    times  = []

    for i in result_path:
        data = sio.loadmat(i)
        times.append(data['time'])
        eps.append(data['epsilon'])

    print(f'Eps {np.mean(eps)}+-{np.std(eps)}')
    print(f'Time {np.mean(times)}+-{np.std(times)}')
    print(f'MAX Time {np.max(times)}')

    print()