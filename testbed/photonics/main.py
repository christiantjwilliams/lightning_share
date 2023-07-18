from lightning_photonics import LightningPhotonics
import matplotlib.pyplot as plt 
import time 
import numpy as np 
import argparse
import os 

# instantiate the object
lightning = LightningPhotonics()

# calibration the system
lightning.calibrate_channels()

# run scalar scalar multiply test
lightning.test_scalar_scalar()

# run vector vector multiply test
lightning.test_vector_vector()

# run real inference
lightning.lenet_mnist_inference()

# get the error distribution
lightning.difference_distribution()

##### Test channel encoding ######
numPts = 20
v = np.ones(20)
out = lightning.upload_and_run(v,np.ones(20))

starting_point = 27
point_pitch = 16

where_to_sample = np.array([starting_point + point_pitch*i for i in range(2*numPts)])
samples = np.array([np.mean(out[starting_point-1+point_pitch*i:starting_point+1+point_pitch*i]) for i in range(2*numPts)])

plt.figure()
plt.plot(out)
plt.plot(where_to_sample,samples,"*")
plt.pause(0.1)
plt.show(block=True)

difference_sampling = samples[1::2] - samples[0::2]
plt.figure()
plt.plot(difference_sampling)
plt.title("Difference Sampling")
plt.show()


####################   Block For Plotting ###########################
plt.show(block=True)
# plt.close("all")

os._exit(0)