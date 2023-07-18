import scipy
from scipy.io import arff
import torch
import torch.nn as nn
import torch.utils.data as data_utils
import pandas as pd
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from lightning_tensorizer import *

train = False

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

device = torch.device('cpu')

# convert the data to tensors
X_train = np.array(six_ft_train_data).astype("int")
y_train = np.array(y_train).astype("int")
X_test = np.array(six_ft_test_data).astype("int")
y_test = np.array(y_test).astype("int")

selected_columns = [
    'duration',
    'protocol_type',
    'src_bytes', 'dst_bytes',
    'count', 'srv_count'
]

size_in_bits = {
    'duration': 16,
    'protocol_type': 8,
    'src_bytes': 16, 'dst_bytes': 16,
    'count': 32, 'srv_count': 32,
}

def binarize_data(Xint):
    Xbin = np.zeros((Xint.shape[0], sum(size_in_bits.values())))
    for i, feature_row in enumerate(Xint):
        #print('i/feature_row', i, feature_row)
        # the index at which the next binary value should be written
        write_ptr = 0
        for j, column_val in enumerate(feature_row):
            #print('j/column_val', j, column_val)
            # Transforming in KB sbytes, dbytes, sload, dload
            if j in [2,3,6,7]:
                column_val = int(column_val/1000) 
            # Setting to maximum any value above the max given the number of b
            if (column_val > 2**size_in_bits[selected_columns[j]] - 1):
                column_val = 2**size_in_bits[selected_columns[j]] - 1
            tmp = list(bin(column_val)[2:])
            tmp = [int(x) for x in tmp]
            # zero padding to the left
            tmp = [0]*(size_in_bits[selected_columns[j]] - len(tmp)) + tmp
            for k, bin_val in enumerate(tmp):
                Xbin[i,write_ptr] = bin_val
                write_ptr += 1
    # BNN dataset
    Xbin[Xbin == 0] = -1
    return Xbin

Xbin = binarize_data(X_train)

print(Xbin.shape)

def convert_list_to_int8(input_list):
    sum = 0
    for i in range(len(input_list)):
        sum += input_list[i] * pow(2, 7-i)
    return int(sum)

Xint8 = []
for item in Xbin:
    temp = []
    for i in range(int(len(Xbin[0])/8)):
        temp.append(convert_list_to_int8(item[i*8:i*8+8]))
    Xint8.append(temp)

X_train = np.array(Xint8)

print(X_train.shape)

Xbin = binarize_data(X_test)

Xint8 = []
for item in Xbin:
    temp = []
    for i in range(int(len(Xbin[0])/8)):
        temp.append(convert_list_to_int8(item[i*8:i*8+8]))
    Xint8.append(temp)

X_test = np.array(Xint8)

X_train, X_test = torch.from_numpy(X_train).to(device), torch.from_numpy(X_test).to(device)
y_train, y_test = torch.from_numpy(y_train).to(device), torch.from_numpy(y_test).to(device)

X_train, X_test = X_train.float(), X_test.float()
y_train, y_test = y_train.long(), y_test.long()

# create a Dataset object for the training data
train_dataset = data_utils.TensorDataset(X_train, y_train)

# create a DataLoader object for the training data
train_dataloader = data_utils.DataLoader(train_dataset, batch_size=10, shuffle=True)

# define the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 12)
        self.fc2 = nn.Linear(12, 6)
        self.fc3 = nn.Linear(6, 3)
        self.fc4 = nn.Linear(3, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        return x

# create an instance of the model
model = Net()
model.to(device)

# define the loss function and optimization algorithm
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

if train:
    # train the model
    for epoch in tqdm(range(100)):
        # loop over the training data
        for X_batch, y_batch in train_dataloader:
            # forward pass
            output = model(X_batch)
            loss = criterion(output, y_batch)

            # backward pass
            loss.backward()
            optimizer.step()

            # zero the gradients
            optimizer.zero_grad()
        torch.save(model, f'checkpoint{epoch}.pt')
        output = model(X_train)
        _, pred = output.max(dim=1)
        correct = pred.eq(y_train).sum().item()
        print(f'checkpoint{epoch} accuracy:', correct/y_train.shape[0])
    
    torch.save(model, 'taurus.pt')
else:
    model = torch.load('taurus.pt')

layer1 = model.fc1.weight.detach().numpy()
layer2 = model.fc2.weight.detach().numpy()
layer3 = model.fc3.weight.detach().numpy()
layer4 = model.fc4.weight.detach().numpy()

print(layer1.shape)
print(layer2.shape)
print(layer3.shape)
print(layer4.shape)

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

# model.fc1.weight = nn.Parameter(torch.FloatTensor(np.reshape(rescale_all_layer[0], (12, 15))))
# model.fc2.weight = nn.Parameter(torch.FloatTensor(np.reshape(rescale_all_layer[1], (6, 12))))
# model.fc3.weight = nn.Parameter(torch.FloatTensor(np.reshape(rescale_all_layer[2], (3, 6))))
# model.fc4.weight = nn.Parameter(torch.FloatTensor(np.reshape(rescale_all_layer[3], (2, 3))))

# int8_model = model

# plt.figure(figsize=(15,3))
# plt.plot(converted_absolute_input_1, label="converted in lightning", marker=".")
# plt.plot(converted_absolute_initial_input_0, label="oracle")
# plt.title("input image")
# plt.legend()
# plt.show()

#output_lightning_1, input_lightning_2, output_lightning_2, input_lightning_3, output_lightning_3 = emulate_lenet_lightning(converted_absolute_input_1, all_converted_absolute_layer, all_converted_sign_layer, 16, scale_factor)
#output_oracle_1, input_oracle_2, output_oracle_2, input_oracle_3, output_oracle_3 = EmulateInference(converted_absolute_input_1, [layer1, layer2, layer3, layer4])
# output_oracle_1_rescale, output_oracle_2_rescale, output_oracle_3_rescale = EmulateInference_rescale(converted_absolute_input_1)

#################
# plt.figure(figsize=(15,3))
# plt.plot(converted_absolute_layer_1, label="converted in lightning", marker=".")
# plt.title("lightning weight layer 1")
# plt.legend()
# plt.show()

# plt.figure(figsize=(15,3))
# plt.plot(np.abs(layer1.reshape((layer1.shape[0]*layer1.shape[1]))), label="oracle absolute", marker=".")
# plt.plot(layer1.reshape((layer1.shape[0]*layer1.shape[1])), label="oracle full")
# plt.title("oracle weight layer 1")
# plt.legend()
# plt.show()

# int_output_oracle_1 = [round(x) for x in output_oracle_1]
# plt.figure(figsize=(15,3))
# plt.plot(output_lightning_1, label="lightning layer 1 output", marker=".")
# plt.plot(int_output_oracle_1, label="oracle layer 1 output", marker=".")
# # plt.plot(output_oracle_1_rescale/scale_factor, label="oracle layer 1 output_rescale", marker=".")
# plt.title("output of layer 1 before relu")
# plt.legend()
# plt.show()

# plt.figure(figsize=(15,3))
# plt.plot(input_lightning_2, label="lightning layer 2 input", marker=".")
# plt.plot(input_oracle_2, label="oracle layer 2 input", marker=".")
# # plt.plot(output_oracle_2_rescale/scale_factor, label="oracle layer 1 output_rescale", marker=".")
# plt.title("output of layer 1 after relu")
# plt.legend()
# plt.show()

# #################
# plt.figure(figsize=(15,3))
# plt.plot(converted_absolute_layer_2, label="converted in lightning", marker=".")
# plt.title("lightning weight layer 2")
# plt.legend()
# plt.show()

# plt.figure(figsize=(15,3))
# plt.plot(np.abs(layer2.reshape((layer2.shape[0]*layer2.shape[1]))), label="oracle absolute", marker=".")
# plt.plot(layer2.reshape((layer2.shape[0]*layer2.shape[1])), label="oracle full")
# plt.title("oracle weight layer 2")
# plt.legend()
# plt.show()

# int_output_oracle_2 = [round(x) for x in output_oracle_2]
# plt.figure(figsize=(15,3))
# plt.plot(output_lightning_2, label="lightning layer 2 output", marker=".")
# plt.plot(int_output_oracle_2, label="oracle layer 2 output", marker=".")
# # plt.plot(output_oracle_3_rescale/scale_factor, label="oracle layer 1 output_rescale", marker=".")
# plt.title("output of layer 2 before relu")
# plt.legend()
# plt.show()

# plt.figure(figsize=(15,3))
# plt.plot(input_lightning_3, label="lightning layer 3 input", marker=".")
# plt.plot(input_oracle_3, label="oracle layer 3 input", marker=".")
# # plt.plot(output_oracle_2_rescale/scale_factor, label="oracle layer 1 output_rescale", marker=".")
# plt.title("output of layer 2 after relu")
# plt.legend()
# plt.show()

# #################
# plt.figure(figsize=(15,3))
# plt.plot(converted_absolute_layer_3, label="converted in lightning", marker=".")
# plt.title("lightning weight layer 3")
# plt.legend()
# plt.show()

# plt.figure(figsize=(15,3))
# plt.plot(np.abs(layer3.reshape((layer3.shape[0]*layer3.shape[1]))), label="oracle absolute", marker=".")
# plt.plot(layer3.reshape((layer3.shape[0]*layer3.shape[1])), label="oracle full")
# plt.title("oracle weight layer 3")
# plt.legend()
# plt.show()

# int_output_oracle_3 = [round(x) for x in output_oracle_3]
# plt.figure(figsize=(15,3))
# plt.plot(output_lightning_3, label="lightning layer 3 output", marker=".")
# plt.plot(int_output_oracle_3, label="oracle layer 3 output", marker=".")
# # plt.plot(output_oracle_3_rescale/scale_factor, label="oracle layer 1 output_rescale", marker=".")
# plt.title("output of layer 3")
# plt.legend()
# plt.show()

# # Convert model to int8, start with float16
# int8_model = torch.load('taurus.pt').half()

# # Convert the model's weights to float16

# # Create a dictionary to store the int8 weights
# int8_weights = {}

# # Determine the maximum absolute value of the float16 weights
# max_weight = max([param.abs().max().item() for name, param in int8_model.named_parameters() if 'weight' in name])

# # Set the scale factor to the maximum value that can be represented in int8
# scale_factor = 255 / max_weight

# # Iterate over the model's weight parameters
# for name, param in model.named_parameters():
#     if 'weight' in name:
#         # Create a tensor with the same shape as the weight tensor, but with dtype int8
#         int8_weights[name] = torch.empty(param.shape, dtype=torch.int8)

#         # Iterate over the float16 weight tensor and copy the scaled and truncated values into the int8 tensor
#         for i, weight in enumerate(param):
#             int8_weights[name][i] = (weight * scale_factor).to(torch.int8)

# # Loop over the int8 weight tensors and copy them to the model
# for name, param in int8_weights.items():
#     int8_model.state_dict()[name].copy_(param)

# # Convert back to float16, so we can run inference (still effectively int8)
#int8_model = int8_model.to(torch.float32)

# evaluate the model on the test data
with torch.no_grad():
    # 32-bit
    # print(X_test.dtype)
    # output = model(X_test)
    # loss = criterion(output, y_test)
    # print("Test loss:", loss.item())
    # _, pred = output.max(dim=1)
    # correct = pred.eq(y_test).sum().item()
    # print('accuracy:', correct/y_test.shape[0])
    # # 16-bit
    model.half()

    X_train = X_train.to(torch.float16)
    output = model(X_train)
    loss = criterion(output, y_train)
    print("Test loss:", loss.item())
    _, pred = output.max(dim=1)
    correct = pred.eq(y_train).sum().item()
    print('accuracy:', correct/y_train.shape[0])

    X_test = X_test.to(torch.float16)
    print(X_test.dtype)
    output = model(X_test)
    loss = criterion(output, y_test)
    print("Test loss:", loss.item())
    _, pred = output.max(dim=1)
    correct = pred.eq(y_test).sum().item()
    print('accuracy:', correct/y_test.shape[0])
    # 8-bit
    # output = int8_model(X_train)
    # loss = criterion(output, y_train)
    # print("Test loss:", loss.item())
    # _, pred = output.max(dim=1)
    # correct = pred.eq(y_train).sum().item()
    # print('accuracy:', correct/y_train.shape[0])

    # output = int8_model(X_test)
    # loss = criterion(output, y_test)
    # print("Test loss:", loss.item())
    # _, pred = output.max(dim=1)
    # correct = pred.eq(y_test).sum().item()
    # print('accuracy:', correct/y_test.shape[0])