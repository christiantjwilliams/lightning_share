import scipy
from scipy.io import arff
import larq as lq
import numpy as np
from tensorflow.python.keras.utils import np_utils
import tensorflow as tf


selected_columns = [
    'duration',
    'protocol_type',
    'src_bytes', 'dst_bytes',
    'count', 'srv_count'
]

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

size_in_bits = {
    'duration': 16,
    'protocol_type': 2,
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

def binarize_labels(Y):
    Y__2_classes = Y.copy()
    # Y__2_classes[Y == 'Normal'] = 0
    # Y__2_classes[Y != 'Normal'] = 1
    Y__2_classes = Y__2_classes.astype('int')
    Y_cat__2_classes = np_utils.to_categorical(Y__2_classes)
    return Y_cat__2_classes

simple_y_test = y_test

X_train = binarize_data(X_train)
X_test = binarize_data(X_test)
y_train = binarize_labels(y_train)
y_test = binarize_labels(y_test)

def build_bnn_model(neurons, 
                    input_shape, 
                    last_act="softmax", 
                    learning_rate=0.0001, 
                    loss='squared_hinge'):
    
    kwargs = dict(input_quantizer="ste_sign",
              kernel_quantizer="ste_sign",
              kernel_constraint="weight_clip")

    model = tf.keras.models.Sequential()
    model.add(lq.layers.QuantDense(neurons[0], use_bias=False,
                                   input_quantizer="ste_sign",
                                   kernel_quantizer="ste_sign",
                                   kernel_constraint="weight_clip",
                                   input_shape=(input_shape,) ) )
    model.add(tf.keras.layers.BatchNormalization(scale=False, momentum=0.9))
    model.add(lq.layers.QuantDense(neurons[1], use_bias=False, **kwargs))
    model.add(tf.keras.layers.BatchNormalization(scale=False, momentum=0.9))
    model.add(lq.layers.QuantDense(neurons[2], use_bias=False, activation=last_act, **kwargs))
    model.add(tf.keras.layers.BatchNormalization(scale=False, momentum=0.9))
    model.add(lq.layers.QuantDense(neurons[3], use_bias=False, activation=last_act, **kwargs))

    # lq.models.summary(model)
    
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
    
    return model

batch_size = 256
train_epochs = 15

neurons = [12, 6, 3, 2]
fold_idx = 0

#tf.print(tf.convert_to_tensor(X_train), summarize=-1)

model = build_bnn_model(neurons, X_train.shape[1])

train_history = model.fit(X_train, y_train, 
                            batch_size=batch_size, 
                            epochs=train_epochs,
                            verbose=1,
                            validation_data=(X_test, y_test))

y_pred = model.predict(X_test)
y_score = predict_classes=np.argmax(y_pred,axis=1)
correct_predictions = np.equal(y_score, simple_y_test)
accuracy = np.mean(correct_predictions)
print('accuracy:', accuracy)
#