import glob

import pickle

import math

import numpy as np
import matplotlib.patches as patches
import matplotlib.cm as cm

if __name__ == "__main__":


    images = []
    # data_train_paths = [f for f in glob.glob('/media/roironen/8AAE21F5AE21DB09/Data/BOMEX_256x256x100_5000CCN_50m_micro_256/train/cloud_*.pkl')]
    data_train_paths = [f for f in glob.glob('/wdata/yaelsc/Roi_ICCP22/CloudCT10sat_polarized_LWC_NOISE/*.pkl')]
    for path in data_train_paths:
        with open(path, 'rb') as outfile:
            x = pickle.load(outfile)
        im = np.concatenate((x['images'],x['DoLPs'][:,None],x['AoLPs'][:,None]),1)

        images.append(im)


    print()