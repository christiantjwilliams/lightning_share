#This file manages experimental equipment and does basic calibration for netcast
import numpy as np 
from tqdm import tqdm
from keysight_wrapper import keysight
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
        self.digitizer = spectrum()
        self.weight_mzi_control = lithium_niobate(1)
        time.sleep(0.2)

        self.dc_readout_task = daq.Task()
        self.dc_readout_task.ai_channels.add_ai_voltage_chan("Dev1/ai0")

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
        oversampling = 10
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

        self.digitizer.start_triggered_acquisition(1)
        self.awg.clear_memory()
        self.awg.upload_waveforms(test_waveforms,markers)
        self.awg.run_awg()
        data = self.digitizer.parse_acquisition_data(1)
        data = np.array(data)


        # print("data shape:", truncated_data.shape)

        return data[front_padding:]

    def upload_and_run_slow(self,ch1_voltage_list):
        #Front pad zero pads the front of the array with a zero and then removes the first element at the end.
        #Clip the voltage ranges to [0,1]
        ch1_voltage_list = np.clip(ch1_voltage_list,self.channel1_low,self.channel1_high)
        digitizer_points_to_acquire = len(ch1_voltage_list)

        test_waveforms = []
        to_send = []
        for i in ch1_voltage_list:
            to_send.append(i*np.ones(self.slow_upload_max_length)) 
        for i in range(16):
            test_waveforms.append(to_send)

        markers = []
        #First channel
        markers.append([10000,0]) #Length, delay
        #Second channel, for digitizer
        markers.append([20,965]) #Length, delay
        markers.append(None)
        markers.append(None)

        self.digitizer.start_triggered_acquisition(digitizer_points_to_acquire)
        self.awg.clear_memory()
        self.awg.upload_waveforms(test_waveforms,markers)
        self.awg.run_awg()
        data = self.digitizer.parse_acquisition_data(digitizer_points_to_acquire)
        data = np.array(data)

        # return np.array([np.mean(data[20+32*i:25+32*i]) for i in range(digitizer_points_to_acquire)])
        return data

    def dc_upload_and_run(self):
        #The point of this method is to measure the bias point when zero voltage is output from the AWG
        run_length = 1
        ch1_voltage_list = np.zeros((run_length))

        test_waveforms = []
        to_send = []
        for i in ch1_voltage_list:
            to_send.append(np.zeros(self.slow_upload_max_length)) 
        for i in range(16):
            test_waveforms.append(to_send)

        markers = []
        #First channel
        markers.append([10000,0]) #Length, delay
        #Second channel, for digitizer
        markers.append([20,50]) #Length, delay
        markers.append(None)
        markers.append(None)

        # self.digitizer.start_triggered_acquisition(digitizer_points_to_acquire)
        self.awg.clear_memory()
        self.awg.upload_waveforms(test_waveforms,markers)
        self.awg.run_awg()
        # data = self.digitizer.parse_acquisition_data(digitizer_points_to_acquire)
        dc_readout_averaging = 10
        readout = 0
        for i in range(dc_readout_averaging):
            readout += self.dc_readout_task.read()/dc_readout_averaging
        return readout

    def find_MZI_bias_point(self,plot=True):
        #Do a course sweep of the bias point to find the minimum valley
        coarse_sweep_num_points = 100
        bias_voltage_sweep = np.linspace(-7,7,num=coarse_sweep_num_points)
        storage = []
        for v in tqdm(bias_voltage_sweep,desc='Finding MZI Bias Point'):
            self.weight_mzi_control.lithium_niobate_biaspoint = v
            self.weight_mzi_control.set_bias_point()
            storage.append(self.dc_upload_and_run())

        storage = np.array(storage)
        #Find the zero point
        bias_point_arg = np.argmin(storage)
        self.mod_bias_point = bias_voltage_sweep[bias_point_arg]
        self.weight_mzi_control.lithium_niobate_biaspoint = self.mod_bias_point
        self.weight_mzi_control.set_bias_point()
        
        if plot == True:
            plt.figure()
            plt.scatter(self.mod_bias_point,storage[bias_point_arg])
            plt.plot(bias_voltage_sweep,storage) #The course sweep
            plt.title("MZI Bias Point Sweep")
            plt.xlabel("Lithium Niobate Bias Voltage")
            plt.ylabel("Optical Power (AU)")
            plt.show(block=False)

    def mod256(self):
        #Here, we will put a linear sweep in voltage onto the modulator and see the response
        voltage_sweep = np.linspace(0,1,num=self.calibration_num_points)
        data = np.zeros(self.calibration_num_points)
        averaging = 150
        for i in tqdm(range(averaging),desc='Generating Calibration'):
            data += self.upload_and_run_slow(voltage_sweep)/averaging
        

        #Create min, max and sweep
        measured_optical = data
        argmin = np.argmin(measured_optical)
        argmax = np.argmax(measured_optical)

        optical_min = measured_optical[argmin]
        optical_max = measured_optical[argmax]

        #Clip data to be between the range of the maximum and minimum
        clipped_voltage = voltage_sweep[argmin:argmax]
        clipped_optical = measured_optical[argmin:argmax]
        if len(clipped_voltage) == 0 or len(clipped_optical) == 0:
            #Have to map the other direction
            clipped_voltage = voltage_sweep[argmax:argmin][::-1]
            clipped_optical = measured_optical[argmax:argmin][::-1]
        
        from numpy.polynomial import Chebyshev
        self.channel_1_mapping_function = np.polynomial.Chebyshev.fit(clipped_optical, clipped_voltage, deg=5, domain=[-1, 1])
        # self.channel_1_mapping_function = np.poly1d(np.polyfit(clipped_optical, clipped_voltage, deg=3))
        self.channel1_range = (optical_min,optical_max)

        #Save all objects and data needed
        object_to_save = [
            self.channel_1_mapping_function,
            self.channel1_range,
            self.mod_bias_point
        ]
        with open('saved_calibration.pkl', 'wb') as file:
           pickle.dump(object_to_save, file)

        fake_optical = np.linspace(self.channel1_range[0],self.channel1_range[1],num=self.calibration_num_points)
        fake_voltage = [self.channel_1_mapping_function(i) for i in fake_optical]

        fake_voltage_sweep = self.upload_and_run_slow(fake_voltage)
        plt.figure()
        plt.plot(voltage_sweep,data)
        plt.plot(fake_voltage,fake_voltage_sweep)
        plt.title("Mod256")
        plt.show(block=False)

        #Sweep through voltage linearly, compare to baseline
        averaging = 50
        intensity_sweep = np.linspace(optical_min,optical_max,num=128)
        voltage_for_precision = [self.channel_1_mapping_function(i) for i in intensity_sweep]
        more_samples = 10
        output = np.zeros(128*more_samples)
        for j in range(more_samples):
            for i in tqdm(range(averaging),"Generating Error"):
                output[j*128:(j+1)*128] += self.upload_and_run_slow(voltage_for_precision)/averaging
        
        tiled_intensity_sweep =  np.tile(intensity_sweep, more_samples)

        error = (tiled_intensity_sweep - output)/optical_max

        dicty = {
            "intensity_sweep":intensity_sweep,
            "output":output,
            "error":error,
            "optical_max":optical_max
        }

        np.save("data_storage/mod_acc/mod_accuracy_" + str(self.timecode) + ".npy",dicty,allow_pickle=True)

        from csv import writer
        list_data=[str(self.timecode),str(np.std(error)),str(np.mean(error))]
        with open('mod_acc.csv', 'a', newline='') as f_object:  
            writer_object = writer(f_object)
            writer_object.writerow(list_data)  
            f_object.close()

        # plt.figure()
        # plt.plot(intensity_sweep)
        # plt.plot(output)
        # plt.legend(["intensity sweep","output"])
        # plt.show(block=False)

        # plt.figure()
        # plt.plot(np.linspace(0,1,num=128),error[:128])
        # plt.xlabel("Encoded Floating Point Value")
        # plt.ylabel("Error")
        # plt.title("Modulator Encoding Error")
        # plt.show(block=False)

        #Error distribution
        # plt.figure(figsize=(8,8))
        # plt.suptitle("Modulator Encoding Error")
        # plt.hist(error, bins=40)
        # plt.title("Difference")
        # plt.xlabel("Difference")
        # plt.ylabel("Frequency")
        # plt.savefig("data_storage/mod_acc/" + str(self.timecode) + ".png",dpi='figure')
        # plt.show(block=False)

    def generate_mod_figure():
        #Once we are parked at the extinction point we output a linear ramp in intensity
        pass

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