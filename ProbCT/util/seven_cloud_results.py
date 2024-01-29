import scipy.io as sio
import numpy as np
import glob, os

relative_error = lambda ext_est, ext_gt, eps=1e-6 : np.linalg.norm(ext_est.ravel() - ext_gt.ravel(),ord=1) / (np.linalg.norm(ext_gt.ravel(),ord=1) + eps)
mass_error = lambda ext_est, ext_gt, eps=1e-6 : (np.linalg.norm(ext_gt.ravel(),ord=1) - np.linalg.norm(ext_est.ravel(),ord=1)) / (np.linalg.norm(ext_gt.ravel(),ord=1) + eps)

paths = [f for f in glob.glob('/wdata/roironen/Data/subset_of_seven_clouds/cloud_results_cloud*')]

eps = []
delta = []
for path in paths:
    file = glob.glob(os.path.join(path, 'logs/*/FINAL_3D_extinction.mat'))[0]
    data = sio.loadmat(file)
    gt = data['GT']
    est = data['extinction']
    eps.append(relative_error(est,gt))
    delta.append(mass_error(est,gt))

print(f'{np.mean(eps)} +- {np.std(eps)}')
print(f'{np.mean(delta)} +- {np.std(delta)}')
print(eps)
print(delta)