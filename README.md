# Learned 3D volumetric recovery of clouds and its uncertainty


## Abstract
Significant uncertainty in climate prediction and cloud physics is tied to observational gaps relating to shallow scattered clouds. 
Addressing these challenges requires remote sensing of their three-dimensional (3D) heterogeneous volumetric scattering content.
This calls for passive scattering computed tomography (CT).
We design a learning-based model (ProbCT) to achieve CT of such clouds, based on noisy multi-view spaceborne images.
ProbCT infers  – for the first time – the posterior probability distribution of the heterogeneous extinction coefficient, per 3D location.
This yields arbitrary valuable statistics, e.g., the 3D field of the most probable extinction and its uncertainty. 
ProbCT uses a neural-field representation, making essentially real-time inference. 
ProbCT undergoes supervised training by a new labeled multi-class database of physics-based volumetric fields of clouds and their corresponding images. We publish this database with the paper. To improve out-of-distribution inference, we incorporate self-supervised learning through differential rendering.
We demonstrate the approach in simulations and on real-world data, and indicate the relevance of 3D recovery and uncertainty to precipitation and renewable energy. 


![Probabalistic_Cloud_tomography](readme_files/tomography.png)

## Description
This repository contains the official implementation of ProbCT model.

![ProbCT](readme_files/framework.png)




&nbsp;


## Installation 
Installation using using anaconda package management

Start a clean virtual environment
```
conda create -n probct python=3.8
source activate probct
```

Install required packages
```
pip install -r requirements.txt
```


&nbsp;


If you use this package in an academic publication please acknowledge the appropriate publications (see LICENSE file). 

