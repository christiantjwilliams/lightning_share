import numpy as np 
from tqdm import tqdm
from keysight_wrapper import keysight
import matplotlib.pyplot as plt 

awg = keysight()

square_wave = np.ones(3000)
square_wave[0:5:10] = -1

#Front pad zero pads the front of the array with a zero and then removes the first element at the end.
#Clip the voltage ranges to [0,1]
square_wave = np.clip(square_wave,-1,1)
#This function automatically applies the compensation technique

test_waveforms = []
for i in range(16):
    test_waveforms.append([square_wave])
markers = []
#First channel
markers.append(None) #Length, delay
#Second channel, for digitizer
markers.append(None) #Length, delay
markers.append(None)
markers.append(None)



for i in range(1000000):
    print(i)
    awg.clear_memory()
    awg.upload_waveforms(test_waveforms,markers)
    awg.run_awg()


plt.close("all")
import os 
os._exit(0)