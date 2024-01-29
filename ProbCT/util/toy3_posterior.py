import copy
import os, glob
import pickle
import matplotlib.pyplot as plt
import numpy as np
from dataloader.noise import SatelliteNoise
import scipy.stats, scipy

def func_p_beta(beta):
    A = 0.75 * (scipy.stats.norm.pdf((beta - 42) * 0.2)) + 0.25 * (scipy.stats.norm.pdf((beta - 78) * 0.2))
    A /= 5
    return A
data_root = '/wdata/roironen/Data'

data_root = os.path.join(data_root, 'Toy_single_voxel_clouds', '10cameras_20m')
data_train_path = [f for f in glob.glob(os.path.join(data_root, "train/cloud*.pkl"))]


images = []
betas = []
for path in data_train_path:
    with open(path, 'rb') as f:
        data = pickle.load(f)

    images.append(data['images'][0])
    betas.append(data['ext'].ravel()[data['index']])

betas = np.array(betas)
images =np.stack(images)
args = np.argsort(betas)
betas = betas[args]
images = images[args]
data_train_path = np.array(data_train_path)[args]


noise = SatelliteNoise(fullwell=13.5e3,bits=10,DARK_NOISE_std=13)
noisy_images = noise.convert_radiance_to_graylevel(images)

y_ind = 10
path = data_train_path[y_ind]
with open(path, 'rb') as f:
    data = pickle.load(f)
y = noisy_images[y_ind]

data2 = copy.deepcopy(data)
data2['images'] = y[None]
with open(path.replace("train","test"), 'wb') as f:
    pickle.dump(data2,f,pickle.HIGHEST_PROTOCOL)


hist = np.histogram(betas, bins=25)
hist_dist = scipy.stats.rv_histogram(hist)

plt.imshow(noisy_images.std(0)[60:100,40:80])
plt.colorbar()
plt.show()

log_p_y_beta_list = []
p_beta = []
for cur_ind in range(len(betas)):

    clean_images = images[cur_ind][60:100,40:80][None]
    noisy_images_curr = noise.convert_radiance_to_graylevel(np.repeat(clean_images,100,axis=0))

    var = noisy_images_curr.std(0)**2
    # log_p_y_beta = -0.5*np.log(var.ravel()).sum() + \
    log_p_y_beta =  (-0.5 * (y[60:100,40:80].ravel() - clean_images.ravel())**2 / (var.ravel())).sum()

    log_p_y_beta_list.append(log_p_y_beta)

    # p_beta.append(hist_dist.pdf(betas[cur_ind]))
    p_beta.append(func_p_beta(betas[cur_ind]))

plt.plot(log_p_y_beta_list)
plt.show()
log_p_y_beta_list=scipy.ndimage.gaussian_filter1d(log_p_y_beta_list,30)
p_beta = np.array(p_beta)
log_p_y_beta_list = np.array(log_p_y_beta_list)
log_p_y_beta_list=log_p_y_beta_list-log_p_y_beta_list.mean()
p_y_beta = np.exp(log_p_y_beta_list)
p_beta /= np.trapz(p_beta, betas)
p_y_beta /= np.trapz(p_y_beta, betas)

posterior = p_y_beta * p_beta
posterior /= np.trapz(posterior, betas)


plt.plot(betas,p_beta)
plt.plot(betas,p_y_beta)
plt.plot(betas,posterior)
plt.legend(['p_beta','p_y_beta','posterior'])
plt.show()



plt.plot(betas,posterior)
plt.hist(betas, bins=25,density=True)
b = np.arange(20,100)
plt.plot(b,hist_dist.pdf(b))
plt.plot(b,func_p_beta(b))
plt.show()




print(betas[y_ind])


