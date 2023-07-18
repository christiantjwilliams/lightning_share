import scipy
from scipy.io import arff
import numpy as np
from lightning_tensorizer import *
import torch

import pickle
from lightning_tensorizer import *

model = torch.load('taurus.pt')
weights = {}
for name, param in model.named_parameters():
    if 'weight' in name:
        weights[name] = param.data.numpy()

weights_array = np.array(weights)

layer1 = weights_array[0]
layer2 = weights_array[1]
layer3 = weights_array[2]
layer4 = weights_array[3]

# load the NSL-KDD dataset from the .arff file
train_data, meta = scipy.io.arff.loadarff('KDDTrain.arff')
test_data, meta = scipy.io.arff.loadarff('KDDTest.arff')
indexes = np.array([0,1,4,5,22,23])
six_ft_train_data = []
six_ft_test_data = []
y_train = []
y_test = []

for pkt in train_data:
    labels_dict = {b'normal':0, b'anomaly':1}
    prot_dict = {b'tcp':0,b'udp':1,b'icmp':2}
    six_ft_train_data.append([pkt[0],prot_dict[pkt[1]],pkt[4],pkt[5],pkt[22],pkt[23]])
    y_train.append(labels_dict[pkt[41]])

for pkt in test_data:
    labels_dict = {b'normal':0, b'anomaly':1}
    prot_dict = {b'tcp':0,b'udp':1,b'icmp':2}
    six_ft_test_data.append([pkt[0],prot_dict[pkt[1]],pkt[4],pkt[5],pkt[22],pkt[23]])
    y_test.append(labels_dict[pkt[41]])

# rescale data to 8+1 bit (-255 ~ +255) scale (FPGA absolute value range)
print("2. Rescale data to map FPGA output range...")
rescale_all_layer, scale_factor = RescaleData([layer1, layer2, layer3, layer4], 8)

# take absolute values and its signs
print("3. Take absolute values and signs...")
absolute_layer_1, sign_layer_1 = TakeAbsoluteValues(rescale_all_layer[0])
absolute_layer_2, sign_layer_2 = TakeAbsoluteValues(rescale_all_layer[1])
absolute_layer_3, sign_layer_3 = TakeAbsoluteValues(rescale_all_layer[2])
absolute_layer_4, sign_layer_4 = TakeAbsoluteValues(rescale_all_layer[3])

print("4. [weight] generate DAC data streams considering 16 samples per cycle...")
converted_absolute_layer_1 = GenerateDataStream(absolute_layer_1, 16, "value")
converted_absolute_layer_2 = GenerateDataStream(absolute_layer_2, 16, "value")
converted_absolute_layer_3 = GenerateDataStream(absolute_layer_3, 16, "value")
converted_absolute_layer_4 = GenerateDataStream(absolute_layer_4, 16, "value")

converted_sign_layer_1 = GenerateDataStream(sign_layer_1, 16, "sign")
converted_sign_layer_2 = GenerateDataStream(sign_layer_2, 16, "sign")
converted_sign_layer_3 = GenerateDataStream(sign_layer_3, 16, "sign")
converted_sign_layer_4 = GenerateDataStream(sign_layer_4, 16, "sign")

all_converted_absolute_layer = [converted_absolute_layer_1, converted_absolute_layer_2, converted_absolute_layer_3, converted_absolute_layer_4]
all_converted_sign_layer = [converted_sign_layer_1, converted_sign_layer_2, converted_sign_layer_3, converted_sign_layer_4]