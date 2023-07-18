#Here is all of the control code for the lithium niobate modulator
import nidaqmx as daq
import numpy as np 

class lithium_niobate():
    def __init__(self,channel):
        if channel == 1:
            device = "Dev1/ao0"
        elif channel == 2:
            device = "Dev1/ao1"
        self.lithium_niobate_biastask = daq.Task()
        self.lithium_niobate_biastask.ao_channels.add_ao_voltage_chan(device)
        self.lithium_niobate_biaspoint = 0 #initial biaspoint

    def set_bias_point(self):
        LiNbO3_bias_point = self.lithium_niobate_biaspoint
        self.lithium_niobate_biastask.write(LiNbO3_bias_point)

    def __del__(self):
        self.lithium_niobate_biastask.close()