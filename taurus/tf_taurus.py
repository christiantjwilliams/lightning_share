from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow_model_optimization as tfmot
import numpy as np
import csv
import scipy
from scipy.io import arff
from sklearn import metrics
import tensorflow as tf
import tensorflow_addons as tfa

train = True

def train_model(model, epochs, trainX, trainY, save=False, write=False, path='.'):

    quantize_model = tfmot.quantization.keras.quantize_model
    q_aware_model = quantize_model(model)

    optimizer = keras.optimizers.Adam(lr=0.0001)
    q_aware_model.compile(optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy'])
    q_aware_model.fit(trainX, trainY, epochs=epochs, batch_size=10, verbose=1, callbacks=[tqdm_callback])

    _, q_aware_model_accuracy = q_aware_model.evaluate(
               trainX, trainY, verbose=0)

    print('Quant train accuracy:', q_aware_model_accuracy)

    return model

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

X_train = np.array(six_ft_train_data).astype("int")
y_train = np.array(y_train).astype("int")
X_test = np.array(six_ft_test_data).astype("int")
y_test = np.array(y_test).astype("int")

# define the model
model = Sequential()
model.add(Dense(12, input_dim=11, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the model to the training data
tqdm_callback = tfa.callbacks.TQDMProgressBar()

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
    'count': 16, 'srv_count': 16,
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

X_train = Xint8

Xbin = binarize_data(X_test)

Xint8 = []
for item in Xbin:
    temp = []
    for i in range(int(len(Xbin[0])/8)):
        temp.append(convert_list_to_int8(item[i*8:i*8+8]))
    Xint8.append(temp)

X_test = Xint8

# X_train_scaled = np.empty_like(X_train)
# X_train_scaled[:] = None

# # Scale each column separately
# for i in range(X_train.shape[1]):
#     col = X_train[:, i]
#     min_val = np.min(col)
#     max_val = np.max(col)
#     col_scaled = (col - min_val) / (max_val - min_val) * 255
#     X_train_scaled[:, i] = col_scaled

# # Convert to Tensor
# X_train = tf.convert_to_tensor(X_train_scaled, dtype=tf.uint8)

# X_test_scaled = np.empty_like(X_test)
# X_test_scaled[:] = None

# # Scale each column separately
# for i in range(X_test.shape[1]):
#     col = X_test[:, i]
#     min_val = np.min(col)
#     max_val = np.max(col)
#     col_scaled = (col - min_val) / (max_val - min_val) * 255
#     X_test_scaled[:, i] = col_scaled

# # Convert to Tensor
# X_test = tf.convert_to_tensor(X_test_scaled, dtype=tf.uint8)

X_train = tf.convert_to_tensor(X_train)
y_train = tf.convert_to_tensor(y_train)
X_test = tf.convert_to_tensor(X_test)
y_test = tf.convert_to_tensor(y_test)

if train:
    model = train_model(model, 100, X_train, y_train)
    model.save("tf_taurus")

else:
    model = keras.models.load_model("tf_taurus")

results = model.evaluate(X_test, y_test, batch_size=10)
print("test loss, test acc:", results)
