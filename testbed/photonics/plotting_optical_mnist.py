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
from scipy.interpolate import interp1d
#### GLOBALS #####
capacitance = 10**(-11)
photon_energy = 1.3e-19
charge_energy = 1.602e-19
energy_per_mac_sweep = np.logspace(-17,-13,num=1000)

#### Fit function imports
dicty = np.load("data_storage/time_integrating_n_1_acc_storage.npy",allow_pickle=True).item()

acc_storage = dicty['acc_storage']
dd_storage = dicty['dd_storage']
s_n1 = interp1d(dd_storage, acc_storage, kind='linear',bounds_error=False)

dicty = np.load("data_storage/time_integrating_n_10_acc_storage.npy",allow_pickle=True).item()

acc_storage = dicty['acc_storage']
dd_storage = dicty['dd_storage']
s_n10 = interp1d(dd_storage, acc_storage, kind='linear',bounds_error=False)

dicty = np.load("data_storage/time_integrating_n_50_acc_storage.npy",allow_pickle=True).item()

acc_storage = dicty['acc_storage']
dd_storage = dicty['dd_storage']
s_n50 = interp1d(dd_storage, acc_storage, kind='linear',bounds_error=False)

N_1_color = "darkorange"
N_10_color = "green"
N_50_color = "red"

plt.figure(figsize=(16,10),dpi=150)

##### N = 1 ###
N = 1
data_N1 = pd.read_csv("optical_mnist_N_1.csv")
data_N1 = data_N1.to_numpy()

#Filter data based on dd error mean
data_N1 = data_N1[np.abs(data_N1[:,5]) < 0.05]

delta_charge_per_MAC_N1 = data_N1[:,0] * 0.2/(2**15) * capacitance/N
delta_energy_per_MAC_N1 = photon_energy/charge_energy * delta_charge_per_MAC_N1
plt.semilogx(delta_energy_per_MAC_N1,data_N1[:,2],"x",color=N_1_color)

##### N = 10 ###
N = 10
data_N10 = pd.read_csv("optical_mnist_N_10.csv")
data_N10 = data_N10.to_numpy()
delta_charge_per_MAC_N10 = data_N10[:,0] * 0.2/(2**15) * capacitance/N
delta_energy_per_MAC_N10 = photon_energy/charge_energy * delta_charge_per_MAC_N10
# plt.semilogx(delta_energy_per_MAC_N10,data_N10[:,2],"x",color=N_10_color)

##### N = 50 ###
N = 50
data_N50 = pd.read_csv("optical_mnist_N_50_200mv.csv")
data_N50 = data_N50.to_numpy()
delta_charge_per_MAC_N50 = data_N50[:,0] * 0.2/(2**15) * capacitance/N
delta_energy_per_MAC_N50 = photon_energy/charge_energy * delta_charge_per_MAC_N50
plt.semilogx(delta_energy_per_MAC_N50,data_N50[:,2],"x",color=N_50_color)

############ Simulation #########
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
plt.semilogx(energy_per_mac_sweep,s_n1(storage),color=N_1_color)


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
# plt.semilogx(energy_per_mac_sweep,s_n10(storage),color=N_10_color)

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
plt.semilogx(energy_per_mac_sweep,s_n50(storage),color=N_50_color)


plt.title("Measured MNIST Accuracy")
plt.ylabel("MNIST Accuracy")
plt.xlabel("Optical Energy per MAC")
plt.legend(["N = 1", "N = 50","Sim N = 1","Sim N = 50"])
plt.show(block=True)