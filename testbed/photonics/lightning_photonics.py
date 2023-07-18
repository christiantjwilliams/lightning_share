import numpy as np 
from tqdm import tqdm
# from keysight_wrapper import keysight
# from spectrum_wrapper import spectrum
import matplotlib.pyplot as plt 
import sys,os,time
from scipy import interpolate
from scipy.io import savemat, loadmat
from scipy.stats import linregress
# import nidaqmx as daq
from lithium_niobate_control import lithium_niobate
import pickle
sys.path.append("../");sys.path.append("../../")
# from hardware_control3.hardware_control3 import agilent_8153a, jdsu_ha9
from lightning_fpga import LightningFPGA
import datetime
from numpy.polynomial import Chebyshev


class LightningPhotonics:
    def __init__(self):
        self.timecode = "{:%Y-%m-%d_%H%M%S}".format(datetime.datetime.now())
        self.dac = LightningFPGA("dac")
        self.adc = LightningFPGA("adc")
        self.weight_mzi_control = lithium_niobate(1)
        self.input_mzi_control = lithium_niobate(2)
        time.sleep(0.2)

        # self.dc_readout_task = daq.Task()
        # self.dc_readout_task.ai_channels.add_ai_voltage_chan("Dev1/ai0")

        self.calibration_num_points = 25

        self.channel1_low = 0
        self.channel1_high = 1
        self.inputmzi_low = 0
        self.inputmzi_high = 1

        # Averaging values
        self.channel_averaging = 16
        self.output_decoder_averaging = 16
        self.generate_output_decoder_averaging = 16
        self.output_decoder_time_integration_calibration_averaging = 10

        self.slow_upload_max_length = 2000 # Minimum of 2000 for keysight AWG
        self.fast_upload_max_length = 5000 # Minimum of 2000 for keysight AWG
        self.bias_offset = -1

        self.print_logo()

    def calibrate_channels(self,plot=True):
        self.find_weightMZI_bias_point(plot=plot) #Bias point is set inside of this function
        self.find_inputMZI_bias_point(plot=plot)
        #The the bias points after sweeping to improve SNR
        self.weight_mzi_control.set_bias_point(); self.input_mzi_control.set_bias_point()
        # plt.show(block=True)
        self.calibrate_channel(plot=plot)
        # plt.show(block=True)
        
        self.calibrate_inputMZI(plot=plot)
        self.calibrate_output_decoder(plot=plot)
        #Verification sweeps
        self.verify_channel(plot=plot)
        self.verify_inputMZI(plot=plot)

        #Save all objects and data needed
        object_to_save = [
            self.channel_1_mapping_function,  # calibrate_channel, first modulator transfer function
            self.channel1_range,
            self.inputMZI_mapping_function,
            self.inputMZI_range,
            self.channel1_output_decoder,
        ]
        with open('saved_calibration.pkl', 'wb') as file:
           pickle.dump(object_to_save, file)

        plt.pause(0.01)

    def load_calibration(self):
        with open('saved_calibration.pkl', 'rb') as file_cal:
            pickel_object = pickle.load(file_cal)
            self.channel_1_mapping_function = pickel_object[0]
            self.channel1_range = pickel_object[1]
            self.inputMZI_mapping_function = pickel_object[2]
            self.inputMZI_range = pickel_object[3]
            self.channel1_output_decoder = pickel_object[4]

    def upload_and_run(self,ch1_voltage_list,LiNbO3_voltage_list):
        #Front pad zero pads the front of the array with a zero and then removes the first element at the end.
        front_padding = 0
        #Clip the voltage ranges to [0,1]
        ch1_voltage_list = np.clip(ch1_voltage_list,self.channel1_low,self.channel1_high)
        LiNbO3_voltage_list = np.clip(LiNbO3_voltage_list,self.inputmzi_low,self.inputmzi_high)
        ch1_voltage_list = np.pad(ch1_voltage_list,(front_padding,0),'edge')
        LiNbO3_voltage_list = np.pad(LiNbO3_voltage_list,(front_padding,0),'edge')
        tempinputs = np.zeros((4*len(ch1_voltage_list)))
        tempweights = np.zeros((4*len(LiNbO3_voltage_list)))
        digitizer_points_to_acquire = len(ch1_voltage_list)
        tempinputs[0::4] = ch1_voltage_list; tempinputs[2::4] = -ch1_voltage_list
        tempweights[0::4] = LiNbO3_voltage_list; tempweights[2::4] = -LiNbO3_voltage_list
        ch1_voltage_list = tempinputs; LiNbO3_voltage_list = tempweights
        #Slow the waveform down, do oversampling
        oversampling = 10
        ch1_voltage_list =  np.repeat(ch1_voltage_list,oversampling)
        LiNbO3_voltage_list =  np.repeat(LiNbO3_voltage_list,oversampling)
        #Pad with zeros until they are > 2000
        if ch1_voltage_list.size < 2000:
            ch1_voltage_list = np.pad(ch1_voltage_list,(0,2000-ch1_voltage_list.size),'constant')
        if LiNbO3_voltage_list.size < 2000:
            LiNbO3_voltage_list = np.pad(LiNbO3_voltage_list,(0,2000-LiNbO3_voltage_list.size),'constant')

        test_waveforms = []
        to_send = [i for i in ch1_voltage_list]
        # for i in ch1_voltage_list:
        #     to_send.append(i*np.ones(self.slow_upload_max_length)) 
        for i in range(15):
            test_waveforms.append([to_send])
        to_send = [i for i in LiNbO3_voltage_list]
        # for i in LiNbO3_voltage_list:
            # to_send.append(i*np.ones(self.slow_upload_max_length))
        test_waveforms.append([to_send])

        markers = []
        #First channel
        markers.append([10000,0]) #Length, delay
        #Second channel, for digitizer
        markers.append([20,50]) #Length, delay
        markers.append(None)
        markers.append(None)

        with open('../../fpga/data/calibration/modulator.txt', 'w') as f:
            f.write(test_waveforms+"\n")
            f.write(markers)
            f.close()

        self.adc.start_triggered_acquisition(1)
        self.dac.run_saved_waveforms(saved_waveforms=1)

        data = self.adc.parse_acquisition_data(1)
        data = np.array(data)

        starting_point = 27
        point_pitch = 16
        where_to_sample = np.array([starting_point + point_pitch*i for i in range(2*digitizer_points_to_acquire)])
        samples = np.array([np.mean(data[starting_point-1+point_pitch*i:starting_point+1+point_pitch*i]) for i in range(2*digitizer_points_to_acquire)])

        truncated_data = samples[1::2] - samples[0::2]
        # print("data shape:", truncated_data.shape)

        return truncated_data[front_padding:]

    def dc_upload_and_run(self):
        #The point of this method is to measure the bias point when zero voltage is output from the AWG
        run_length = 1
        mod1_voltage_list = np.zeros((run_length))
        mod2_voltage_list = np.zeros((run_length))

        digitizer_points_to_acquire = len(mod1_voltage_list)
        test_waveforms = []
        to_send = []
        for i in mod1_voltage_list:
            to_send.append(i*np.ones(self.slow_upload_max_length)) 
        test_waveforms.append(to_send)

        to_send = []
        for i in mod2_voltage_list:
            to_send.append(i*np.ones(self.slow_upload_max_length))
        test_waveforms.append(to_send)

        markers = []
        #First channel
        markers.append([10000,0]) #Length, delay
        #Second channel, for digitizer
        markers.append([20,50]) #Length, delay
        markers.append(None)
        markers.append(None)

        print("test_waveforms", test_waveforms)
        print("markers", markers)

        with open('../../fpga/data/calibration/dc_modulator.txt', 'w') as f:
            f.write(test_waveforms+"\n")
            f.write(markers)
            f.close()

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

    def calibrate_channel(self,plot=True):
        #Here we sweep the electro-optic voltage on a channel and measure the transfer function
        voltage_sweep = np.linspace(self.channel1_low,self.channel1_high,num=self.calibration_num_points)
        temp = 0
        for a in tqdm(range(self.channel_averaging),desc="Calibrating Weight MZI"):
            temp += self.upload_and_run(voltage_sweep,np.ones(self.calibration_num_points))/self.channel_averaging
        #Run it a couple of times
        storage = np.array(temp)
        self.channel1_voltage_sweep = voltage_sweep
        self.channel1_initial_calibration = storage

        #Now we will fit a polynomial to the modulator transfer function
        self.channel_1_mapping_function, self.channel1_range = self.generate_mapping_function(voltage_sweep,storage)
        fake_optical = np.linspace(self.channel1_range[0],self.channel1_range[1],num=self.calibration_num_points)
        fake_voltage = [self.channel_1_mapping_function(i) for i in fake_optical]
        time.sleep(0.1)
        data = self.upload_and_run(fake_voltage, np.ones(self.calibration_num_points))
        if plot == True:
            plt.figure()
            plt.plot(voltage_sweep,storage)
            plt.plot(fake_voltage,data)
            plt.title("Weight MZI Calibration")
            plt.legend(["Original Sweep","Fit"])
            plt.show(block=False)

    def verify_channel(self, plot=True):
        print("Verifying Channel Calibration")
        #Now we will fit a polynomial to the modulator transfer function
        fake_optical = np.linspace(self.channel1_range[0],self.channel1_range[1],num=self.calibration_num_points)
        fake_voltage = [self.channel_1_mapping_function(i) for i in fake_optical]
        time.sleep(0.1)
        data = self.upload_and_run(fake_voltage, np.ones(self.calibration_num_points))
        if plot == True:
            plt.figure()
            plt.plot(self.channel1_voltage_sweep,self.channel1_initial_calibration)
            plt.plot(fake_voltage,data)
            plt.title("Weight MZI Verification")
            plt.legend(["Original Sweep","Fit"])
            plt.show(block=False)

    def calibrate_inputMZI(self,plot=True):
        #Here we sweep the electro-optic voltage on a channel and measure the transfer function
        voltage_sweep = np.linspace(self.inputmzi_low,self.inputmzi_high,num=self.calibration_num_points)
        temp = 0
        for i in tqdm(range(self.channel_averaging),desc="Calibrating Input MZI"):
            temp += self.upload_and_run(np.ones(self.calibration_num_points),voltage_sweep)/self.channel_averaging
        #Run it a couple of times
        storage = np.array(temp)
        self.inputMZI_voltage_sweep = voltage_sweep
        self.inputMZI_initial_calibration = storage

        #Now we will fit a polynomial to the modulator transfer function
        self.inputMZI_mapping_function, self.inputMZI_range = self.generate_mapping_function(voltage_sweep,storage)
        fake_optical = np.linspace(self.inputMZI_range[0],self.inputMZI_range[1],num=self.calibration_num_points)
        fake_voltage = [self.inputMZI_mapping_function(i) for i in fake_optical]
        time.sleep(0.1)
        data = self.upload_and_run(np.ones(self.calibration_num_points),fake_voltage)
        
        if plot == True:
            plt.figure()
            plt.plot(voltage_sweep,storage)
            plt.plot(fake_voltage,data)
            plt.title("InputMZI Calibration")
            plt.legend(["Original Sweep","Fit"])
            plt.show(block=False) 

    def verify_inputMZI(self,plot=True):
        print("Verifying InputMZI Calibration")
        fake_optical = np.linspace(self.inputMZI_range[0],self.inputMZI_range[1],num=self.calibration_num_points)
        fake_voltage = [self.inputMZI_mapping_function(i) for i in fake_optical]
        time.sleep(0.1)
        data = self.upload_and_run(np.ones(self.calibration_num_points),fake_voltage)
        if plot == True:
            plt.figure()
            plt.plot(self.inputMZI_voltage_sweep,self.inputMZI_initial_calibration)
            plt.plot(fake_voltage,data)
            plt.title("InputMZI Calibration")
            plt.legend(["Original Sweep","Fit"])
            plt.show(block=False) 

    def calibrate_output_decoder(self, plot=True):
        self.channel1_output_decoder = self.generate_output_decoder()
        #Verify this, send a sweep into both modulators, detect the output and decode it
        sweep = np.linspace(0,1,num=self.calibration_num_points)
        ch1_volt_to_send = np.array([self.channel_1_mapping_function(
            self.floats_to_channel_range(i, self.channel1_range)) for i in sweep])
        inputMZI_to_send = np.array([self.inputMZI_mapping_function(
            self.floats_to_channel_range(i, self.inputMZI_range)) for i in sweep])
        output = np.zeros(len(ch1_volt_to_send))
        
        for a in range(self.output_decoder_averaging):
            output += self.upload_and_run(ch1_volt_to_send,inputMZI_to_send)/self.output_decoder_averaging
        fit = [i**2 for i in sweep]
        #Decode the output
        decoded = self.channel1_output_decoder(output) #Decoded are the estimated floating point values

        #Create a polynomial mapping between the output decoded value and the optical power
        from numpy.polynomial import Chebyshev
        t = np.linspace(0,1,num=25)

        if plot == True:
            plt.figure()
            plt.plot(decoded)
            plt.plot(fit)
            plt.legend(["Calibration","Ideal"])
            plt.title("Channel 1 Output Decoder Calibration")
            plt.xlabel("Timestep (AU)")
            plt.ylabel("Optical Intensity (AU)")
            plt.show(block=False)
            
    def generate_output_decoder(self):
        print("Generating Output Decoder")
        #To generate an output decoder function we are going to send in a "0" and then send in a "1" on both calibrated modulators
        num_sample = 5
        ch1_volt_to_send = self.channel_1_mapping_function(
            self.floats_to_channel_range(1, self.channel1_range))
        inputMZI_to_send = self.inputMZI_mapping_function(
            self.floats_to_channel_range(1, self.inputMZI_range))
        one_times_one = 0
        for a in tqdm(range(self.generate_output_decoder_averaging),desc="One Times One Case"):
            one_times_one += np.median(self.upload_and_run(
                ch1_volt_to_send*np.ones(num_sample), inputMZI_to_send*np.ones(num_sample)))/self.generate_output_decoder_averaging

        ch1_volt_to_send = self.channel_1_mapping_function(
            self.floats_to_channel_range(0, self.channel1_range))
        inputMZI_to_send = self.inputMZI_mapping_function(
            self.floats_to_channel_range(0, self.inputMZI_range))
        zero_times_zero = 0
        for a in tqdm(range(self.generate_output_decoder_averaging),desc="Zero Times Zero Case"):
            zero_times_zero += np.median(self.upload_and_run(
                ch1_volt_to_send*np.ones(num_sample), inputMZI_to_send*np.ones(num_sample)))/self.generate_output_decoder_averaging
        self.zero_times_zero = zero_times_zero
        f = interpolate.interp1d([zero_times_zero, one_times_one], [
            0, 1], bounds_error=False, fill_value='extrapolate')
        return f


    def floats_to_channel_range(self, float_value, channel_range, floatmin=0, floatmax=1):
        slope = (channel_range[1] - channel_range[0])/(floatmax - floatmin)
        y_intercept = channel_range[1] - slope * floatmax
        return float_value * slope + y_intercept

    def find_inputMZI_bias_point(self,plot=True):
        #Do a course sweep of the bias point to find the minimum valley
        coarse_sweep_num_points = 50
        bias_voltage_sweep = np.linspace(-9,9,num=coarse_sweep_num_points)
        storage = []
        for v in tqdm(bias_voltage_sweep,desc='Finding Input MZI Bias Point'):
            self.input_mzi_control.lithium_niobate_biaspoint = v
            self.input_mzi_control.set_bias_point()
            storage.append(self.dc_upload_and_run())

        storage = np.array(storage)
        #Find the zero point
        bias_point_arg = np.argmin(storage) + self.bias_offset
        self.input_mzi_control.lithium_niobate_biaspoint = bias_voltage_sweep[bias_point_arg]
        # self.input_mzi_control.set_bias_point()
        
        if plot == True:
            plt.figure()
            plt.scatter(bias_voltage_sweep[bias_point_arg],storage[bias_point_arg])
            plt.plot(bias_voltage_sweep,storage) #The course sweep
            plt.title("Input MZI Bias Point Sweep")
            plt.xlabel("Lithium Niobate Bias Voltage")
            plt.ylabel("Optical Power (AU)")
            plt.show(block=False)

    def find_weightMZI_bias_point(self,plot=True):
        # Do a course sweep of the bias point to find the minimum valley
        coarse_sweep_num_points = 50
        bias_voltage_sweep = np.linspace(-3,3,num=coarse_sweep_num_points)
        storage = []
        for v in tqdm(bias_voltage_sweep,desc='Finding Weight MZI Bias Point'):
            self.weight_mzi_control.lithium_niobate_biaspoint = v
            self.weight_mzi_control.set_bias_point()
            storage.append(self.dc_upload_and_run())
        #Find minimum valley
        storage = np.array(storage)
        bias_point_arg = np.argmin(storage) + self.bias_offset

        self.weight_mzi_control.lithium_niobate_biaspoint = bias_voltage_sweep[bias_point_arg]
        # self.weight_mzi_control.set_bias_point()
        
        if plot == True:
            plt.figure()
            plt.scatter(bias_voltage_sweep[bias_point_arg],storage[bias_point_arg])
            plt.plot(bias_voltage_sweep,storage) #The course sweep
            plt.title("Weight MZI Bias Point Sweep")
            plt.xlabel("Lithium Niobate Bias Voltage")
            plt.ylabel("Optical Power (AU)")
            plt.show(block=False)

    def generate_mapping_function(self, applied_voltage, measured_optical):
        #First we want to isolate the range we should work in.
        argmin = np.argmin(measured_optical)
        argmax = np.argmax(measured_optical)

        optical_min = measured_optical[argmin]
        optical_max = measured_optical[argmax]

        #Clip data to be between the range of the maximum and minimum
        clipped_voltage = applied_voltage[argmin:argmax]
        clipped_optical = measured_optical[argmin:argmax]
        if len(clipped_voltage) == 0:
            #Have to map the other direction
            clipped_voltage = applied_voltage[argmax:argmin][::-1]
            clipped_optical = measured_optical[argmax:argmin][::-1]
        
        
        # f = np.polynomial.Chebyshev.fit(
        #     clipped_optical, clipped_voltage, deg=3, domain=[-1, 1])
        f = np.poly1d(np.polyfit(clipped_optical, clipped_voltage, deg=3))

        return f, (optical_min, optical_max)

    def slow_float_upload(self,ch1_values,input_values):
        arraych1 = np.array([self.channel_1_mapping_function(
                self.floats_to_channel_range(x, self.channel1_range)) for x in ch1_values])
        arrayinputMZI = np.array([self.inputMZI_mapping_function(
                self.floats_to_channel_range(x, self.inputMZI_range)) for x in input_values])
        output = self.upload_and_run(arraych1, arrayinputMZI)
        output = np.array([self.channel1_output_decoder(x) for x in output])
        return output

    def difference_distribution(self, plot=True):
        #We are going to do MANY elementwise products and get a distribution of their error
        # print("Running Difference Distribution With Time Integration from nn")
        self.sum_receiver_code = 0  # We will start "counting photons"
        num_runs = 100
        run_length = 30
        difference_storage = []
        #Randomly create data
        x1s = np.random.rand(num_runs,run_length)
        x2s = np.random.rand(num_runs,run_length)
        
        gt_storage = []
        optical_storage = []
        for i in tqdm(range(num_runs), desc="Diffference Distribution no time integration"):
            x1 = x1s[i,:]
            x2 = x2s[i,:]
            ground_truth_product = np.multiply(x1, x2)
            gt_storage.append(ground_truth_product)
            
            measured_value = self.slow_float_upload(x1, x2)
            optical_storage.append(measured_value)
            difference = measured_value - ground_truth_product
            difference_storage.append(difference)

        #Calculate optical energy per MAC
        difference_storage = np.array(difference_storage).flatten()
        optical_storage = np.array(optical_storage).flatten()
        gt_storage = np.array(gt_storage).flatten()
        self.optical_storage = optical_storage
        self.gt_storage = gt_storage

        average_digitizer_code = self.sum_receiver_code/(num_runs*run_length)

        compute_mean = np.mean(difference_storage)
        compute_std = np.std(difference_storage)

        #Write into the CSV
        from csv import writer
        list_data=[str(average_digitizer_code),str(compute_mean),str(compute_std)]
        with open('SNR_measurement_N_1.csv', 'a', newline='') as f_object:  
            writer_object = writer(f_object)
            writer_object.writerow(list_data)  
            f_object.close()

        self.sum_receiver_code = 0
        if plot == True:
            num_bins = 200
            plt.figure(figsize=(20,8))
            plt.suptitle("Difference Distribution")
            plt.subplot(131)
            plt.hist(difference_storage, bins=num_bins)
            plt.title("Difference")
            plt.xlabel("Difference")
            plt.ylabel("Frequency")

            plt.subplot(132)
            plt.hist(gt_storage, bins=num_bins)
            plt.title("Ground Truth")
            plt.xlabel("Ground Truth")
            plt.ylabel("Frequency")

            plt.subplot(133)
            plt.hist(optical_storage, bins=num_bins)
            plt.title("Optical")
            plt.xlabel("Optical")
            plt.ylabel("Frequency")
            plt.show(block=False)
        return difference_storage

    def vector_vector_inner_product_slow(self, vector1, vector2):
        #Here we are going to handle a vector vector inner product
        assert len(vector1) == len(vector2)
        #Convert float to voltage
        #We are going to put the weight on channel 1 and the input on the inputMZI

        #For use with AWG
        measured_product = self.slow_float_upload(
            np.abs(vector1), np.abs(vector2))

        sign_corrected = np.zeros_like(measured_product)
        for i in range(len(vector1)):
            if vector1[i] >= 0 and vector2[i] >= 0:
                sign_corrected[i] = measured_product[i]
            elif vector1[i] >= 0 and vector2[i] < 0:
                sign_corrected[i] = -measured_product[i]
            elif vector1[i] < 0 and vector2[i] >= 0:
                sign_corrected[i] = -measured_product[i]
            elif vector1[i] < 0 and vector2[i] < 0:
                sign_corrected[i] = measured_product[i]
            else:
                print("PROBLEM!")

        #Do the accumulation step
        measured_product = np.sum(sign_corrected)
        return measured_product

    def matrix_matrix_slow(self,matrix1,matrix2):
        first_dim = matrix1.shape[0]
        common_dimension = matrix1.shape[1]
        output_dim = matrix2.shape[1]

        zero_full = np.zeros(common_dimension)

        measured_product = np.zeros((first_dim, output_dim))
        scale1 = np.max(np.abs(matrix1))
        scale2 = np.max(np.abs(matrix2))

        #Now we have generated the voltage we step through the values
        for i in tqdm(range(first_dim)):
            for k in range(output_dim):
                ch1 = self.vector_vector_inner_product_slow(
                    matrix2[:, k]/scale2, matrix1[i, :]/scale1)
                measured_product[i, k] = ch1
        measured_product = measured_product * scale1 * scale2
        return measured_product

    def matrix_matrix_fast(self,matrix1,matrix2,disable_tqdm=False):
        first_dim = matrix1.shape[0]
        common_dimension = matrix1.shape[1]
        output_dim = matrix2.shape[1]

        zero_full = np.zeros(common_dimension)

        measured_product = np.zeros((first_dim, output_dim))
        scale1 = np.max(np.abs(matrix1))
        scale2 = np.max(np.abs(matrix2))
        
        def relu(x):
            return x * (x > 0)

        #Now we have generated the voltage we step through the values
        for i in tqdm(range(first_dim),desc="Matrix_matrix Product Fast",disable=disable_tqdm):
            for k in range(output_dim):
                pospos = self.vector_vector_inner_product_fast(
                    relu(matrix2[:, k]/scale2), relu(matrix1[i, :]/scale1))
                negneg = self.vector_vector_inner_product_fast(
                    relu(-matrix2[:, k]/scale2), relu(-matrix1[i, :]/scale1))
                posneg = self.vector_vector_inner_product_fast(
                    relu(matrix2[:, k]/scale2), relu(-matrix1[i, :]/scale1))
                negpos = self.vector_vector_inner_product_fast(
                    relu(-matrix2[:, k]/scale2), relu(matrix1[i, :]/scale1))
                measured_product[i, k] = pospos + negneg - negpos - posneg

        measured_product = measured_product * scale1 * scale2
        return measured_product

    def test_matrix_matrix_slow(self):
        #This is a very lightweight test that just makes sure the scheduler works...
        non_common_dim = 8
        common_dimension = 100
        input_matrix = 3*np.random.rand(non_common_dim, common_dimension) - 1.5
        weight_matrix = 3*np.random.rand(common_dimension, non_common_dim) - 1.5
        product_matrix = np.dot(input_matrix, weight_matrix)

        measured_product = self.matrix_matrix_slow(input_matrix,weight_matrix)

        fig,axs = plt.subplots(1,3,figsize=(12,9))
        fig.suptitle("Test Matrix Matrix Slow")
        pos = axs[0].imshow(product_matrix)
        axs[0].set_title("Standard Product")
        fig.colorbar(pos)

        pos = axs[1].imshow(measured_product)
        axs[1].set_title("Optically Computed Product")
        fig.colorbar(pos)

        pos = axs[2].imshow(np.abs(product_matrix - measured_product))
        axs[2].set_title("Difference Matrix")
        fig.colorbar(pos)
        plt.pause(0.001)

    def test_matrix_matrix_fast(self):
        #This is a very lightweight test that just makes sure the scheduler works...
        non_common_dim = 8
        common_dimension = 70
        input_matrix = 3*np.random.rand(non_common_dim, common_dimension) - 1.5
        weight_matrix = 3*np.random.rand(common_dimension, non_common_dim) - 1.5
        product_matrix = np.dot(input_matrix, weight_matrix)

        measured_product = self.matrix_matrix_fast(input_matrix,weight_matrix)

        fig,axs = plt.subplots(1,3,figsize=(12,9))
        fig.suptitle("Test Matrix Matrix Fast")
        pos = axs[0].imshow(product_matrix)
        axs[0].set_title("Standard Product")
        fig.colorbar(pos)

        pos = axs[1].imshow(measured_product)
        axs[1].set_title("Optically Computed Product")
        fig.colorbar(pos)

        pos = axs[2].imshow(np.abs(product_matrix - measured_product))
        axs[1].set_title("Difference Matrix")
        fig.colorbar(pos)
        plt.pause(0.001)

    def test_vector_vector_inner_product_slow(self):
        #Test matricies which are in the range of -0.5 to 0.5 rather than 0 to 1
        #Now we do a matrix matrix product down the channel
        matrix_dim = 8
        input_matrix = np.random.rand(matrix_dim, matrix_dim)
        weight_matrix = np.random.rand(matrix_dim, matrix_dim)
        product_matrix = np.dot(input_matrix, weight_matrix)

        measured_product = np.zeros((matrix_dim, matrix_dim))

        #Now we have generated the voltage we step through the values
        for i in tqdm(range(matrix_dim)):
            for k in range(matrix_dim):
                measured_product[i, k] = self.vector_vector_inner_product_slow(
                    weight_matrix[:, k], input_matrix[i, :])

        fig = plt.figure()
        pos = plt.imshow(product_matrix)
        plt.title("Vector-Vector Standard Product")
        fig.colorbar(pos)

        fig = plt.figure()
        pos = plt.imshow(measured_product)
        plt.title("Vector-Vector Optically Computed Product")
        fig.colorbar(pos)

        fig = plt.figure()
        pos = plt.imshow(np.abs(product_matrix - measured_product))
        plt.title("Vector-Vector Difference Matrix")
        fig.colorbar(pos)
        plt.pause(0.001)

    def test_vector_vector(self):
        pass

    def test_scalar_scalar(self):
        print("Running Test of Elementwise Scalar Product")
        #Generate two vectors for computation
        num_samples = 50
        input_vec = np.random.rand(num_samples)
        weight_vec = np.random.rand(num_samples)
        product_vec = [input_vec[i]*weight_vec[i] for i in range(num_samples)]
        #We are going to put the weight on channel 1 and the input on the inputMZI

        #For use with AWG
        measured_product = self.slow_float_upload(
            weight_vec, input_vec)

        plt.figure()
        plt.plot(product_vec)
        plt.plot(measured_product)
        plt.xlabel("Random Product Index")
        plt.ylabel("Product")
        plt.title("Testing Random Products")
        plt.legend(["Generated Digitally", "Generated Optically"])
        plt.show(block=False)

    def lenet_mnist_inference(self):
        pass

    def two_layer_mnist_slow(self,save_plot=False):
        dicty = np.load("noise_robust_mnist_weights.npy",allow_pickle=True).item()
        layer1_weights = dicty["layer1"]
        layer2_weights = dicty["layer2"]
        layer3_weights = dicty['layer3']
        bias1 = dicty['b1']
        bias2 = dicty['b2']
        bias3 = dicty['b3']
        xtest = dicty["xtest"]
        ytest = dicty["ytest"]

        def relu(vec):
            return vec * (vec > 0)

        first_layer_activations = np.dot(xtest, layer1_weights) + bias1
        #Apply relu
        first_layer_activations = relu(first_layer_activations)
        second_layer_activations = np.dot(first_layer_activations, layer2_weights) + bias2
        second_layer_activations = relu(second_layer_activations)

        output_activations = np.dot(second_layer_activations,layer3_weights) + bias3

        classification_values = np.argmax(output_activations, 1)
        num_correct = 0
        num_total = 0
        for i in range(classification_values.shape[0]):
            if classification_values[i] == np.argmax(ytest[i]):
                num_correct += 1
            num_total += 1

        baseline_accuracy = num_correct/num_total

        print("Baseline Model Accuracy:", baseline_accuracy)

        batch_size = 100
        xtest_optical = xtest[:batch_size, :]
        ytest_optical = ytest[:batch_size]
        print("input shape: ",xtest_optical.shape)
        print("layer 1 shape: ", layer1_weights.shape)
        print("layer 2 shape: ", layer2_weights.shape)
        print("output shape: ", ytest_optical.shape)
        ############ FIRST LAYER ###############
        print("Starting First Layer")
        first_layer_activations = np.dot(xtest_optical, layer1_weights) + bias1
        first_layer_activations = relu(first_layer_activations)
        second_layer_activations = np.dot(first_layer_activations,layer2_weights) + bias2
        second_layer_activations = relu(second_layer_activations)
        self.digitizer_codes_mnist = []
        optical_output_activations = self.matrix_matrix_slow(second_layer_activations, layer3_weights) + bias3

        #Save a bunch of relevent data if we need it later
        to_save_dict = {
            "digitizer_codes_mnist" : self.digitizer_codes_mnist,
            "optical_output_activations":optical_output_activations,
            "optical_difference_distribution":self.optical_storage,
            "ground_truth_difference_distribution":self.gt_storage,
            "zero_times_zero":self.zero_times_zero,
            "vector_length_zero_intercept":self.vector_length_to_zero_intercept
        }

        #Difference distribution std dev
        dd_std_dev = np.std(self.optical_storage - self.gt_storage)

        filename = "data_storage/N_1_mnist/_" + str(self.timecode) + ".npy"
        np.save(filename,to_save_dict,allow_pickle=True)
        
        #Infer mean optical energy 
        average_digitizer_code = np.mean(np.array(self.digitizer_codes_mnist))
        
        classification_values = np.argmax(optical_output_activations, 1)
        num_correct = 0
        num_total = 0
        for i in range(classification_values.shape[0]):
            if classification_values[i] == np.argmax(ytest[i]):
                num_correct += 1
            num_total += 1

        optical_accuracy = num_correct/num_total

        #Write into the CSV
        from csv import writer
        list_data=[str(average_digitizer_code),str(baseline_accuracy),str(optical_accuracy),str(dd_std_dev),str(self.timecode)]
        with open('optical_mnist_N_1.csv', 'a', newline='') as f_object:  
            writer_object = writer(f_object)
            writer_object.writerow(list_data)  
            f_object.close()

        print("Optically Computed Accuracy:", optical_accuracy)
        return optical_accuracy

    def two_layer_mnist_fast(self,N=10,save_plot=False):
        dicty = np.load("noise_robust_mnist_weights.npy",allow_pickle=True).item()
        layer1_weights = dicty["layer1"]
        layer2_weights = dicty["layer2"]
        layer3_weights = dicty['layer3']
        bias1 = dicty['b1']
        bias2 = dicty['b2']
        bias3 = dicty['b3']
        xtest = dicty["xtest"]
        ytest = dicty["ytest"]

        def relu(vec):
            return vec * (vec > 0)

        first_layer_activations = np.dot(xtest, layer1_weights) + bias1
        #Apply relu
        first_layer_activations = relu(first_layer_activations)
        second_layer_activations = np.dot(first_layer_activations, layer2_weights) + bias2
        second_layer_activations = relu(second_layer_activations)

        output_activations = np.dot(second_layer_activations,layer3_weights) + bias3

        classification_values = np.argmax(output_activations, 1)
        num_correct = 0
        num_total = 0
        for i in range(classification_values.shape[0]):
            if classification_values[i] == np.argmax(ytest[i]):
                num_correct += 1
            num_total += 1

        baseline_accuracy = num_correct/num_total

        print("Baseline Model Accuracy:", baseline_accuracy)

        batch_size = 100
        xtest_optical = xtest[:batch_size, :]
        ytest_optical = ytest[:batch_size]
        ############ FIRST LAYER ###############
        print("Starting First Layer")
        first_layer_activations = np.dot(xtest_optical, layer1_weights) + bias1
        first_layer_activations = relu(first_layer_activations)
        second_layer_activations = np.dot(first_layer_activations,layer2_weights) + bias2
        second_layer_activations = relu(second_layer_activations)
        self.digitizer_codes_mnist = []
        
        num_partition_steps = second_layer_activations.shape[1]//N
        optical_output_activations = np.zeros((second_layer_activations.shape[0],layer3_weights.shape[1]))
        for i in tqdm(range(num_partition_steps),desc="Number of Partitions for MNIST N= " + str(N) + " is " + str(num_partition_steps)):
            optical_output_activations += self.matrix_matrix_fast(second_layer_activations[:,int(i*N):int((i+1)*N)], layer3_weights[int(i*N):int((i+1)*N),:],disable_tqdm=True) + bias3

        if N == 10:
            temp_optical_storage = self.optical_storage_N_10
            temp_gt_storage = self.gt_storage_N_10
        if N == 50:
            temp_optical_storage = self.optical_storage_N_50
            temp_gt_storage = self.gt_storage_N_50
        if N == 100:
            temp_optical_storage = self.optical_storage_N_100
            temp_gt_storage = self.gt_storage_N_100
        #Save a bunch of relevent data if we need it later
        to_save_dict = {
            "digitizer_codes_mnist" : self.digitizer_codes_mnist,
            "optical_output_activations":optical_output_activations,
            "optical_difference_distribution":temp_optical_storage,
            "ground_truth_difference_distribution":temp_gt_storage,
            "zero_times_zero":self.zero_times_zero,
            "vector_length_zero_intercept":self.vector_length_to_zero_intercept
        }

        #Difference distribution std dev
        dd_std_dev = np.std(temp_optical_storage - temp_gt_storage)

        filename = "data_storage/N_" + str(N) + "_mnist/_" + str(self.timecode) + ".npy"
        np.save(filename,to_save_dict,allow_pickle=True)
        
        #Infer mean optical energy 
        average_digitizer_code = np.mean(np.array(self.digitizer_codes_mnist))
        
        classification_values = np.argmax(optical_output_activations, 1)
        num_correct = 0
        num_total = 0
        for i in range(classification_values.shape[0]):
            if classification_values[i] == np.argmax(ytest[i]):
                num_correct += 1
            num_total += 1

        optical_accuracy = num_correct/num_total

        #Write into the CSV
        from csv import writer
        list_data=[str(average_digitizer_code),str(baseline_accuracy),str(optical_accuracy),str(dd_std_dev),str(self.timecode)]
        with open("optical_mnist_N_" + str(N) + ".csv", 'a', newline='') as f_object:  
            writer_object = writer(f_object)
            writer_object.writerow(list_data)  
            f_object.close()

        print("Optically Computed Accuracy:", optical_accuracy)
        return optical_accuracy

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
        os._exit(0)