from pandas.core.algorithms import diff
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../");sys.path.append("../../");sys.path.append("../../../");sys.path.append("../../../../")
plt.style.use("../../../../matplotlib_style.mplstyle")
import seaborn as sns
sns.set()
from tqdm import tqdm
import pandas as pd 
from scipy import interpolate
from scipy.interpolate import interp1d

#### GLOBALS
photon_energy = 1.3e-19
charge_energy = 1.602e-19
voltage_per_code = 0.2/2**(15)
femto_gain = 1
capacitance = 1e-11
mean_threshold_n_1 = 0.05
mean_threshold_n_10 = 0.5
mean_threshold_n_50 = 0.7
N_1_color = "darkorange"
N_10_color = "green"
N_50_color = "red"
energy_per_mac_sweep = np.logspace(-17,-13,num=1000)


plt.figure(figsize=(10,6))
################ N = 1 case #############
data = pd.read_csv("SNR_measurement_N_1.csv")
data = data.to_numpy()

data = data[np.abs(data[:,1]) < mean_threshold_n_1]

average_delta_code_per_mac = data[:,0]


average_delta_voltage_per_mac = voltage_per_code * average_delta_code_per_mac/femto_gain
average_charge_per_mac = capacitance* average_delta_voltage_per_mac
average_optical_energy_per_mac = photon_energy/charge_energy * average_charge_per_mac
plt.semilogx(average_optical_energy_per_mac,data[:,2],"x")


############## N = 10 case ###########
data = pd.read_csv("SNR_measurement_N_10.csv")
data = data.to_numpy()

data = data[np.abs(data[:,1]) < mean_threshold_n_10]

average_delta_code_per_mac = data[:,0]
average_delta_voltage_per_mac = voltage_per_code * average_delta_code_per_mac/femto_gain
average_charge_per_mac = capacitance* average_delta_voltage_per_mac
average_optical_energy_per_mac = photon_energy/charge_energy * average_charge_per_mac
# plt.semilogx(average_optical_energy_per_mac,data[:,2],"x",color="red")

############## N = 50 case ###########
data = pd.read_csv("SNR_measurement_N_50.csv")
data = data.to_numpy()

data = data[np.abs(data[:,1]) < mean_threshold_n_50]

average_delta_code_per_mac = data[:,0]
average_delta_voltage_per_mac = voltage_per_code * average_delta_code_per_mac/femto_gain
average_charge_per_mac = capacitance* average_delta_voltage_per_mac
average_optical_energy_per_mac = photon_energy/charge_energy * average_charge_per_mac
# plt.semilogx(average_optical_energy_per_mac,data[:,2],"x",color="green")

################# Fundamental noise curves ####################
boltzmann_constant = 1.3e-23
T = 300
charge_noise = np.sqrt(boltzmann_constant*T*capacitance)
#We will simulate it here
num_samples = 10000
inp = np.random.rand(num_samples)
weight = np.random.rand(num_samples)

output = np.dot(inp, weight)
SNR_storage = []
storage = []
rms_voltage_noise = 300e-6
charge_noise = rms_voltage_noise * capacitance
for e in tqdm(energy_per_mac_sweep):
    charge_per_mac = charge_energy/photon_energy * e
    SNR = charge_per_mac/(charge_noise)
    SNR_storage.append(SNR)
    optical_output = output + 1./SNR * (np.random.rand(num_samples) - 0.5)
    difference = output - optical_output
    storage.append(np.std(difference))
plt.semilogx(energy_per_mac_sweep,storage)


plt.ylim(0,1)
plt.title("Time Integrating Receiver Energy per MAC")
plt.ylabel("Difference Distribution Std Dev")
plt.xlabel("Optical Energy per MAC")
plt.legend(["Experimental","Simulation"])
# plt.savefig("../figures/LiNbO3_slow_TIA_difference_distribution_std_dev.png")
plt.show(block=False)

################ CONVERT TO ACCURACY ##################

plt.figure(figsize=(10,6))
################ N = 1 case #############
dicty = np.load("data_storage/time_integrating_n_1_acc_storage.npy",allow_pickle=True).item()

acc_storage = dicty['acc_storage']
dd_storage = dicty['dd_storage']
s_n1 = interp1d(dd_storage, acc_storage, kind='linear',bounds_error=False)


data = pd.read_csv("SNR_measurement_N_1.csv")
data = data.to_numpy()

data = data[np.abs(data[:,1]) < mean_threshold_n_1]

average_delta_code_per_mac = data[:,0]

average_delta_voltage_per_mac = voltage_per_code * average_delta_code_per_mac/femto_gain
average_charge_per_mac = capacitance* average_delta_voltage_per_mac
average_optical_energy_per_mac = photon_energy/charge_energy * average_charge_per_mac
plt.semilogx(average_optical_energy_per_mac,s_n1(data[:,2]),"x",color=N_1_color)


############## N = 10 case ###########
dicty = np.load("data_storage/time_integrating_n_10_acc_storage.npy",allow_pickle=True).item()

acc_storage = dicty['acc_storage']
dd_storage = dicty['dd_storage']
s_n10 = interp1d(dd_storage, acc_storage, kind='linear',bounds_error=False)

data = pd.read_csv("SNR_measurement_N_10.csv")
data = data.to_numpy()

data = data[np.abs(data[:,1]) < mean_threshold_n_10]

average_delta_code_per_mac = data[:,0]
average_delta_voltage_per_mac = voltage_per_code * average_delta_code_per_mac/femto_gain
average_charge_per_mac = capacitance * average_delta_voltage_per_mac
average_optical_energy_per_mac = photon_energy/charge_energy * average_charge_per_mac
plt.semilogx(average_optical_energy_per_mac,s_n10(data[:,2]),"x",color=N_10_color)

############### N = 50 case ###########
dicty = np.load("data_storage/time_integrating_n_50_acc_storage.npy",allow_pickle=True).item()

acc_storage = dicty['acc_storage']
dd_storage = dicty['dd_storage']
s_n50 = interp1d(dd_storage, acc_storage, kind='linear',bounds_error=False)

data = pd.read_csv("SNR_measurement_N_50.csv")
data = data.to_numpy()

data = data[np.abs(data[:,1]) < mean_threshold_n_50]

average_delta_code_per_mac = data[:,0]
average_delta_voltage_per_mac = voltage_per_code * average_delta_code_per_mac/femto_gain
average_charge_per_mac = capacitance* average_delta_voltage_per_mac
average_optical_energy_per_mac = photon_energy/charge_energy * average_charge_per_mac
plt.semilogx(average_optical_energy_per_mac,s_n50(data[:,2]),"x",color=N_50_color)

################# Fundamental noise curves ####################
charge_noise = np.sqrt(boltzmann_constant*T*capacitance)
#We will simulate it here

#### N = 1 #########
num_samples = 10000
inp = np.random.rand(num_samples)
weight = np.random.rand(num_samples)

output = np.dot(inp, weight)
SNR_storage = []
storage = []
rms_voltage_noise = 300e-6
charge_noise = capacitance*rms_voltage_noise
for e in tqdm(energy_per_mac_sweep):
    charge_per_mac = charge_energy/photon_energy * e
    SNR = charge_per_mac/(charge_noise)
    SNR_storage.append(SNR)
    optical_output = output + 1./SNR * (np.random.rand(num_samples) - 0.5)
    difference = output - optical_output
    storage.append(np.std(difference))
plt.semilogx(energy_per_mac_sweep,s_n1(storage),N_1_color)


#### N = 10 #########
num_samples = 10000
N = 10
inp = np.random.rand(num_samples,N)
weight = np.random.rand(num_samples,N)

output = np.multiply(inp,weight)
output = np.sum(output,axis=1)
SNR_storage = []
storage = []
for e in tqdm(energy_per_mac_sweep):
    charge_per_mac = charge_energy/photon_energy * e
    SNR = charge_per_mac/(charge_noise)
    SNR_storage.append(SNR)
    optical_output = output + 1./SNR * (np.random.rand(num_samples) - 0.5)
    difference = output - optical_output
    storage.append(np.std(difference))
plt.semilogx(energy_per_mac_sweep,s_n10(storage),N_10_color)

#### N = 50 #########
num_samples = 10000
N = 50
inp = np.random.rand(num_samples,N)
weight = np.random.rand(num_samples,N)

output = np.multiply(inp,weight)
output = np.sum(output,axis=1)
SNR_storage = []
storage = []
for e in tqdm(energy_per_mac_sweep):
    charge_per_mac = charge_energy/photon_energy * e
    SNR = charge_per_mac/(charge_noise)
    SNR_storage.append(SNR)
    optical_output = output + 1./SNR * (np.random.rand(num_samples) - 0.5)
    difference = output - optical_output
    storage.append(np.std(difference))
plt.semilogx(energy_per_mac_sweep,s_n50(storage),N_50_color)

plt.ylim(0,1)
plt.xlim(1e-17,1e-13)
plt.title("MNIST Accuracy Time Integrating Receiver")
plt.ylabel("Accuracy")
plt.xlabel("Optical Energy per MAC")
plt.legend(["Experimental N = 1","Experimental N = 10","Experimental N = 50","Simulation N = 1", "Simulation N = 10","Simulation N = 50"])
plt.show(block=True)