import time 
import matplotlib.pyplot as plt
from modulator_acc import in_network_inference 

ioi = in_network_inference()
ioi.find_MZI_bias_point()
time.sleep(0.2)
ioi.mod256()
plt.close("all")
import os 
os._exit(0)