#This file manages experimental equipment and does basic calibration for netcast
from turtle import pu
import numpy as np 
from tqdm import tqdm
from keysight_1GSPS_wrapper import keysight
import matplotlib.pyplot as plt 
from spectrum_wrapper import spectrum
import sys,os,time
from scipy import interpolate
from scipy.io import savemat, loadmat
from scipy.stats import linregress
import nidaqmx as daq
from lithium_niobate_control import lithium_niobate
import pickle
sys.path.append("../");sys.path.append("../../")
from hardware_control3.hardware_control3 import agilent_8153a, jdsu_ha9
import datetime

class in_network_inference:
    def __init__(self):
        self.timecode = "{:%Y-%m-%d_%H%M%S}".format(datetime.datetime.now())
        # self.powermeter = agilent_8153a.Agilent8153A(address='GPIB0::23::INSTR')
        self.awg = keysight()
        self.weight_mzi_control = lithium_niobate(1)
        time.sleep(0.2)

        # self.dc_readout_task = daq.Task()
        # self.dc_readout_task.ai_channels.add_ai_voltage_chan("Dev1/ai0")

        self.calibration_num_points = 100

        self.channel1_low = 0
        self.channel1_high = 1

        self.inputmzi_low = 0
        self.inputmzi_high = 1

        #Averaging values
        self.channel_averaging = 16
        self.output_decoder_averaging = 16
        self.generate_output_decoder_averaging = 16

        self.slow_upload_max_length = 2000 #Minimum of 2000 for keysight AWG
        self.fast_upload_max_length = 5000 #Minimum of 2000 for keysight AWG

        self.print_logo()

    def upload_and_run(self,ch1_voltage_list):
        #Front pad zero pads the front of the array with a zero and then removes the first element at the end.
        front_padding = 0
        #Clip the voltage ranges to [0,1]
        ch1_voltage_list = np.clip(ch1_voltage_list,self.channel1_low,self.channel1_high)
        ch1_voltage_list = np.pad(ch1_voltage_list,(front_padding,0),'edge')
        digitizer_points_to_acquire = len(ch1_voltage_list)
        #Slow the waveform down, do oversampling
        oversampling = 1
        ch1_voltage_list =  np.repeat(ch1_voltage_list,oversampling)
        #Pad with zeros until they are > 2000
        if ch1_voltage_list.size < 2000:
            ch1_voltage_list = np.pad(ch1_voltage_list,(0,2000-ch1_voltage_list.size),'constant')
        
        test_waveforms = []
        to_send = [i for i in ch1_voltage_list]
        # for i in ch1_voltage_list:
        #     to_send.append(i*np.ones(self.slow_upload_max_length)) 
        for i in range(16):
            test_waveforms.append([to_send])

        markers = []
        #First channel
        markers.append([10000,0]) #Length, delay
        #Second channel, for digitizer
        markers.append([20,50]) #Length, delay
        markers.append(None)
        markers.append(None)

        self.awg.clear_memory()
        self.awg.upload_waveforms(test_waveforms,markers)
        self.awg.run_awg()

    def generate_mod_figure(self):
        #Load in the calibration, repeatedly run the pulse sweep
        #Put the pulse sweep into generator function
        with open('saved_calibration.pkl', 'rb') as file_cal:
            pickel_object = pickle.load(file_cal)
            self.channel_1_mapping_function = pickel_object[0]
            self.channel1_range = pickel_object[1]
            self.mod_bias_point = pickel_object[2]

        self.weight_mzi_control.lithium_niobate_biaspoint = self.mod_bias_point
        self.weight_mzi_control.set_bias_point()

        pulse_sweep = np.linspace(self.channel1_range[0],self.channel1_range[1],num=256)
        pulse_sweep = np.repeat(pulse_sweep,2)
        pulse_sweep[1::2] = 0
        converted_pulse_sweep = np.array([self.channel_1_mapping_function(i) for i in pulse_sweep])
        print("Fast mod outputing")
        while True:
            self.upload_and_run(converted_pulse_sweep)
        
    def mnist_latency_figure(self):
        dicty = np.load("neural_networks/noise_robust_mnist_weights.npy",allow_pickle=True).item()
        layer1_weights = dicty["layer1"]
        layer2_weights = dicty["layer2"]
        layer3_weights = dicty['layer3']
        bias1 = dicty['b1']
        bias2 = dicty['b2']
        bias3 = dicty['b3']
        xtest = dicty["xtest"]
        ytest = dicty["ytest"]

        ind = 0
        print(xtest.shape)
        image_to_send = xtest[ind,:]
        image_to_send[-1] = 255
        # self.upload_and_run()
        
        # plt.figure()
        # plt.imshow(xtest[ind,:].reshape((28,28)))
        # plt.show(block=True)
        while True:
            self.upload_and_run(image_to_send)

    def latency_measurement(self):
        vector_length = 10000
        while True:
            self.upload_and_run(np.ones(vector_length))

    def print_logo(self):
        try:
            a_file = open("../banner.txt")
        except:
            a_file = open("banner.txt")
        lines = a_file.readlines()
        for line in lines:
            print(line[:-2])
        a_file.close()

    def __del__(self):
        import os 
        os._exit(0)

if __name__ == "__main__":
    ioi = in_network_inference()
    # gen_mod_figure = ioi.generate_mod_figure()
    ioi.latency_measurement()