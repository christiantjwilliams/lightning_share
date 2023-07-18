"""
laserResonanceSweep.py
Configure laser to sweep; trigger NI-DAQ on TTL laser output
C. Panuski 06JUN19 (D-Day 75 Anniversary)
"""

import sys
sys.path.append("C:\\Users\PhC-Setup\Documents\GitHub\hardware_control\hardware_control")
import time
import datetime
import numpy as np

currTime = "{:%Y-%m-%d_%H%M%S}".format(datetime.datetime.now())

import matplotlib.pyplot as plt 
import scipy.io as sio
from scipy.optimize import *

from .santectsl710 import SantecTSL710
import nidaqmx as daq
from nidaqmx.constants import AcquisitionType

### Configure laser & sweep parameters ###
laser = SantecTSL710("GPIB0::29")
startWav = laser.get_wavelength();
lamStart = 1518.4
lamStop = 1519.2
pwr = 0
speed = 1 #nm/s
trigStep = .002 #output trigger pulse ever X nm
laser.enable_output_trigger(trigStep=trigStep)
scanMode = 1 # one-way continuous scan
nSweeps = 1
tSweep = (lamStop-lamStart)/speed
wav = np.arange(lamStart,lamStop+trigStep,trigStep)

### Configure NI-DAQ ###
# Start a task for analog sampling based on external clock
vSamp = daq.Task()
# Add analog input from AI0 named "PD input" win min and max voltages
vSamp.ai_channels.add_ai_voltage_chan("Dev1/ai2","PD Input",min_val=0,max_val=1)
# Set clock timing; use default rising edge sampling
fSamp = 5000 # From NI documentation, set fSamp to fastest rate expected from external clk
sourcePin = "PFI0"
nBuffer = 25000 #max number of samples to be collected
vSamp.timing.cfg_samp_clk_timing(fSamp,source=sourcePin,
        sample_mode=AcquisitionType.CONTINUOUS,samps_per_chan=nBuffer)

### Laser sweep
vSamp.start() #start AI sampling

for i in range(nSweeps):    
    laser.wavelength_sweep(lamStart,lamStop,pwr,scanMode,speed) #start sweep
    # Use blocking read -- don't continue until ALL samples have been collected
    vPD = vSamp.read(number_of_samples_per_channel=np.size(wav),timeout=tSweep+10)
    print((np.size(vPD)))
    
    # Fit plot and save data
    plt.plot(wav,vPD,'-x')
    plt.show()

    sio.savemat('laserSweep_'+currTime,{'wav':wav, 'vPD':vPD})
    
vSamp.stop()
laser.set_wavelength(startWav)
