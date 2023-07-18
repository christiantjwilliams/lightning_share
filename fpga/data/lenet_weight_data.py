import pickle
import numpy as np


## for 14-bit coding, 0x7FFC is 8191, 0x8000 is -8192
def tohex(val, nbits):
  return hex((val + (1 << nbits)) % (1 << nbits))


## input a list of decimal integer, output a list of hex (16 bits) strings
def Dec2Hex(dec_streams):
    hex_list = []
    hex_stream = ""
    for i in range(len(dec_streams)):
        hex_value = tohex(dec_streams[i]*4, 16)
        if len(hex_value) < 6:
            hex_value = '0x' + '0'*(6-len(hex_value)) + hex_value[2-len(hex_value):]
        # hex_streams.append(bytes(hex_value, encoding="raw_unicode_escape"))
        hex_list.append(hex_value)
        hex_stream += hex_value[2:]

        hex_stream = str(hex_stream)

    hex_stream = hex_stream.upper()

    return hex_list, hex_stream


def SeparateNegPos(layer):
    positive_matrix = np.zeros((layer.shape[0], layer.shape[1]))
    negative_matrix = np.zeros((layer.shape[0], layer.shape[1]))

    for i in range(layer.shape[0]):
        for j in range(layer.shape[1]):
            if layer[i,j] > 0:
                positive_matrix[i,j] = layer[i,j]
            
            if layer[i,j] < 0:
                negative_matrix[i,j] = layer[i,j]

    return positive_matrix, negative_matrix


def TakeAbsoluteValues(layer):
    absolute_matrix = np.zeros((layer.shape[0], layer.shape[1]))
    sign_matrix = np.zeros((layer.shape[0], layer.shape[1]))

    for i in range(layer.shape[0]):
        for j in range(layer.shape[1]):
            if layer[i,j] > 0:
                absolute_matrix[i,j] = layer[i,j]
                sign_matrix[i,j] = 1
            
            if layer[i,j] < 0:
                absolute_matrix[i,j] = -layer[i,j]
                sign_matrix[i,j] = 0

    return absolute_matrix, sign_matrix


def RescaleData(multiple_layers):
    maxvalue = 0
    minvalue = 999

    global_maxvalue = 0

    max_fpga = 8191

    for layer in multiple_layers:
        if maxvalue < np.max(layer):
            maxvalue = np.max(layer)
        if minvalue > np.min(layer):
            minvalue = np.min(layer)

    global_maxavlue = max(maxvalue, -minvalue)

    scale_factor = max_fpga / global_maxavlue

    rescale_multiple_layer = []
    for layer in multiple_layers:
        rescale_layer = np.zeros((layer.shape[0], layer.shape[1]))
        for i in range(layer.shape[0]):
            for j in range(layer.shape[1]):
                if layer[i, j] != 0:
                    rescale_layer[i,j] = int(layer[i, j] * scale_factor)

        rescale_multiple_layer.append(rescale_layer)

    return rescale_multiple_layer


def PreserveDataConvert(data, process_sign):
    # converted_data = str(width) + "'h"
    converted_data = []
    nonzero_indices = []
    integrating_lengths = []
    datalength = 0
    indexlength = 0

    if process_sign == "positive":
        for i in range(data.shape[0]):
            row_nonzero_indices = []
            for j in range(data.shape[1]):
                if data[i,j] >= 0:  
                    converted_data.append(int(data[i,j]))
                    converted_data.append(int(data[i,j]))  # downgrade DAC speed by a factor of 2
                    row_nonzero_indices.append(j)
                    datalength += 2
                
                if data[i,j] < 0:
                    converted_data.append(0)
                    converted_data.append(0)  # downgrade DAC speed by a factor of 2
                    row_nonzero_indices.append(j)
                    datalength += 2

            indexlength += 1
            nonzero_indices.append(row_nonzero_indices)
            integrating_lengths.append(len(row_nonzero_indices))

    elif process_sign == "negative":
        for i in range(data.shape[0]):
            row_nonzero_indices = []
            for j in range(data.shape[1]):
                if data[i,j] >= 0:  
                    converted_data.append(0)
                    converted_data.append(0)  # downgrade DAC speed by a factor of 2
                    row_nonzero_indices.append(j)
                    datalength += 2
                
                if data[i,j] < 0:
                    converted_data.append(-int(data[i,j]))
                    converted_data.append(-int(data[i,j]))  # downgrade DAC speed by a factor of 2
                    row_nonzero_indices.append(j)
                    datalength += 2

            indexlength += 1
            nonzero_indices.append(row_nonzero_indices)
            integrating_lengths.append(len(row_nonzero_indices))
        
    if datalength % 16 > 0:
        for i in range(16 - (datalength % 16)):
            converted_data.append(0)
    
    # print(indexlength)
    if indexlength % 16 > 0:
        for i in range(16 - (indexlength % 16)):
            integrating_lengths.append(0)

    return converted_data, nonzero_indices, integrating_lengths


def DataConvert(data):
    # converted_data = str(width) + "'h"
    converted_data = []
    nonzero_indices = []
    integrating_lengths = []
    datalength = 0
    indexlength = 0

    for i in range(data.shape[0]):
        row_nonzero_indices = []
        for j in range(data.shape[1]):
            if data[i,j] > 0:
                converted_data.append(int(data[i,j]))
                converted_data.append(int(data[i,j]))  # downgrade DAC speed by a factor of 2
                row_nonzero_indices.append(j)
                datalength += 2
            
            if data[i,j] < 0:
                converted_data.append(int(-data[i,j]))
                converted_data.append(int(-data[i,j]))  # downgrade DAC speed by a factor of 2
                row_nonzero_indices.append(j)
                datalength += 2

        indexlength += 1
        nonzero_indices.append(row_nonzero_indices)
        integrating_lengths.append(len(row_nonzero_indices))
        
    if datalength % 16 > 0:
        for i in range(16 - (datalength % 16)):
            converted_data.append(0)
    
    # print(indexlength)
    if indexlength % 16 > 0:
        for i in range(16 - (indexlength % 16)):
            integrating_lengths.append(0)

    return converted_data, nonzero_indices, integrating_lengths


if __name__ == "__main__":
    # MNIST LeNet Model
    # layer1 = pickle.load(open("/home/zhizhenz/lightning-master/data/saved_models/lenet/compressed_fc_1.p", "rb"))
    # layer2 = pickle.load(open("/home/zhizhenz/lightning-master/data/saved_models/lenet/compressed_fc_2.p", "rb"))
    # layer3 = pickle.load(open("/home/zhizhenz/lightning-master/data/saved_models/lenet/compressed_fc_3.p", "rb"))

    layer1 = pickle.load(open("/home/zhizhenz/lightning-master/data/saved_models/lenet/fc_1.p", "rb"))
    layer2 = pickle.load(open("/home/zhizhenz/lightning-master/data/saved_models/lenet/fc_2.p", "rb"))
    layer3 = pickle.load(open("/home/zhizhenz/lightning-master/data/saved_models/lenet/fc_3.p", "rb"))

    # bias1 = pickle.load(open("/home/zhizhenz/lightning-master/data/saved_models/lenet/compressed_bias_1.p", "rb"))
    # bias2 = pickle.load(open("/home/zhizhenz/lightning-master/data/saved_models/lenet/compressed_bias_2.p", "rb"))
    # bias3 = pickle.load(open("/home/zhizhenz/lightning-master/data/saved_models/lenet/compressed_bias_3.p", "rb"))

    # bias1_r = bias1.reshape((1, bias1.shape[0]))
    # bias2_r = bias2.reshape((1, bias2.shape[0]))
    # bias3_r = bias3.reshape((1, bias3.shape[0]))

    # rescale data to 0 - 8191 scale
    # rescale_multiple_layers = RescaleData([layer1, layer2, layer3, bias1_r, bias2_r, bias3_r])
    rescale_multiple_layers = RescaleData([layer1, layer2, layer3])

    # separate positive and negative matrices
    # positive_layer_1, negative_layer_1 = SeparateNegPos(rescale_multiple_layers[0])
    # positive_layer_2, negative_layer_2 = SeparateNegPos(rescale_multiple_layers[1])
    # positive_layer_3, negative_layer_3 = SeparateNegPos(rescale_multiple_layers[2])

    # take absolute values and its signs
    absolute_layer_1, sign_layer_1 = TakeAbsoluteValues(rescale_multiple_layers[0])
    absolute_layer_2, sign_layer_2 = TakeAbsoluteValues(rescale_multiple_layers[1])
    absolute_layer_3, sign_layer_3 = TakeAbsoluteValues(rescale_multiple_layers[2])

    # mark nonzero matrices and convert full matrix to positive data streams list
    # converted_positive_data_1, positive_indices_1, positive_integrating_lengths_1 = PreserveDataConvert(positive_layer_1, "positive")
    # converted_positive_data_2, positive_indices_2, positive_integrating_lengths_2 = PreserveDataConvert(positive_layer_2, "positive")
    # converted_positive_data_3, positive_indices_3, positive_integrating_lengths_3 = PreserveDataConvert(positive_layer_3, "positive")
    # converted_negative_data_1, negative_indices_1, negative_integrating_lengths_1 = PreserveDataConvert(negative_layer_1, "negative")
    # converted_negative_data_2, negative_indices_2, negative_integrating_lengths_2 = PreserveDataConvert(negative_layer_2, "negative")
    # converted_negative_data_3, negative_indices_3, negative_integrating_lengths_3 = PreserveDataConvert(negative_layer_3, "negative")

    converted_absolute_data_1, absolute_indices_1, absolute_integrating_lengths_1 = PreserveDataConvert(absolute_layer_1, "positive")
    converted_absolute_data_2, absolute_indices_2, absolute_integrating_lengths_2 = PreserveDataConvert(absolute_layer_2, "positive")
    converted_absolute_data_3, absolute_indices_3, absolute_integrating_lengths_3 = PreserveDataConvert(absolute_layer_3, "positive")


    # # store the converted weights in pickle
    # pickle.dump(converted_positive_data_1, open("lenet/converted_positive_data_1.p", "wb"))
    # pickle.dump(converted_positive_data_2, open("lenet/converted_positive_data_2.p", "wb"))
    # pickle.dump(converted_positive_data_3, open("lenet/converted_positive_data_3.p", "wb"))
    # pickle.dump(converted_negative_data_1, open("lenet/converted_negative_data_1.p", "wb"))
    # pickle.dump(converted_negative_data_2, open("lenet/converted_negative_data_2.p", "wb"))
    # pickle.dump(converted_negative_data_3, open("lenet/converted_negative_data_3.p", "wb"))

    # # store the nonzero indices of weights in pickle
    # pickle.dump(positive_indices_1, open("lenet/positive_indices_1.p", "wb"))
    # pickle.dump(positive_indices_2, open("lenet/positive_indices_2.p", "wb"))
    # pickle.dump(positive_indices_3, open("lenet/positive_indices_3.p", "wb"))
    # pickle.dump(negative_indices_1, open("lenet/negative_indices_1.p", "wb"))
    # pickle.dump(negative_indices_2, open("lenet/negative_indices_2.p", "wb"))
    # pickle.dump(negative_indices_3, open("lenet/negative_indices_3.p", "wb"))

    # # store the integration length of each row in the weight matrix in pickle
    # pickle.dump(positive_integrating_lengths_1, open("lenet/positive_integrating_lengths_1.p", "wb"))
    # pickle.dump(positive_integrating_lengths_2, open("lenet/positive_integrating_lengths_2.p", "wb"))
    # pickle.dump(positive_integrating_lengths_3, open("lenet/positive_integrating_lengths_3.p", "wb"))
    # pickle.dump(negative_integrating_lengths_1, open("lenet/negative_integrating_lengths_1.p", "wb"))
    # pickle.dump(negative_integrating_lengths_2, open("lenet/negative_integrating_lengths_2.p", "wb"))
    # pickle.dump(negative_integrating_lengths_3, open("lenet/negative_integrating_lengths_3.p", "wb"))

    # get hex streams from int
    # positive_hex_list_1, positive_hex_stream_1 = Dec2Hex(converted_positive_data_1)
    # positive_hex_list_2, positive_hex_stream_2 = Dec2Hex(converted_positive_data_2)
    # positive_hex_list_3, positive_hex_stream_3 = Dec2Hex(converted_positive_data_3)
    # negative_hex_list_1, negative_hex_stream_1 = Dec2Hex(converted_negative_data_1)
    # negative_hex_list_2, negative_hex_stream_2 = Dec2Hex(converted_negative_data_2)
    # negative_hex_list_3, negative_hex_stream_3 = Dec2Hex(converted_negative_data_3)

    absolute_hex_list_1, absolute_hex_stream_1 = Dec2Hex(converted_absolute_data_1)
    absolute_hex_list_2, absolute_hex_stream_2 = Dec2Hex(converted_absolute_data_2)
    absolute_hex_list_3, absolute_hex_stream_3 = Dec2Hex(converted_absolute_data_3)

    # bia_list_1 = []
    # counter_1 = 0
    # for x in rescale_multiple_layers[3].reshape(bias1.shape):
    #     counter_1 += 1
    #     bia_list_1.append(int(x))
    # if counter_1 % 16 > 0:
    #     for i in range(16 - (counter_1 % 16)):
    #         bia_list_1.append(0)

    # bia_list_2 = []
    # counter_2 = 0
    # for x in rescale_multiple_layers[4].reshape(bias2.shape):
    #     counter_2 += 1
    #     bia_list_2.append(int(x))
    # if counter_2 % 16 > 0:
    #     for i in range(16 - (counter_2 % 16)):
    #         bia_list_2.append(0)
    
    # bia_list_3 = []
    # counter_3 = 0
    # for x in rescale_multiple_layers[5].reshape(bias3.shape):
    #     counter_3 += 1
    #     bia_list_3.append(int(x))
    # if counter_3 % 16 > 0:
    #     for i in range(16 - (counter_3 % 16)):
    #         bia_list_3.append(0)
        
    # bias_hex_list_1, bias_hex_stream_1 = Dec2Hex(bia_list_1)
    # bias_hex_list_2, bias_hex_stream_2 = Dec2Hex(bia_list_2)
    # bias_hex_list_3, bias_hex_stream_3 = Dec2Hex(bia_list_3)

    ## write SRAM
    sram_addr = 0
    with open('lenet/absolute_hex_stream_1.txt', 'w') as f:
        f.write(absolute_hex_stream_1)
        f.write("\n" + str(len(absolute_hex_stream_1)*4))
    with open('lenet/sram_absolute_hex_stream_1.txt', 'w+') as f:
        for k in range(int(len(absolute_hex_stream_1)/64)):
            f.write("init_data[{}] = 256'h{};\n".format(sram_addr, absolute_hex_stream_1[64*k: 64*k+64]))
            sram_addr += 1

    with open('lenet/absolute_hex_stream_2.txt', 'w') as f:
        f.write(absolute_hex_stream_2)
        f.write("\n" + str(len(absolute_hex_stream_2)*4))
    with open('lenet/sram_absolute_hex_stream_2.txt', 'w+') as f:
        for k in range(int(len(absolute_hex_stream_2)/64)):
            f.write("init_data[{}] = 256'h{};\n".format(sram_addr, absolute_hex_stream_2[64*k: 64*k+64]))
            sram_addr += 1

    with open('lenet/absolute_hex_stream_3.txt', 'w') as f:
        f.write(absolute_hex_stream_3)
        f.write("\n" + str(len(absolute_hex_stream_3)*4))
    with open('lenet/sram_absolute_hex_stream_3.txt', 'w+') as f:
        for k in range(int(len(absolute_hex_stream_3)/64)):
            f.write("init_data[{}] = 256'h{};\n".format(sram_addr, absolute_hex_stream_3[64*k: 64*k+64]))
            sram_addr += 1

    vector_sign_layer_1 = list(sign_layer_1.reshape((sign_layer_1.shape[0]*sign_layer_1.shape[1])))
    vector_sign_layer_2 = list(sign_layer_2.reshape((sign_layer_2.shape[0]*sign_layer_2.shape[1])))
    vector_sign_layer_3 = list(sign_layer_3.reshape((sign_layer_3.shape[0]*sign_layer_3.shape[1])))

    str_vector_sign_layer_1 = ''
    for i in vector_sign_layer_1:
        str_vector_sign_layer_1 += str(int(i))

    str_vector_sign_layer_2 = ''
    for i in vector_sign_layer_2:
        str_vector_sign_layer_2 += str(int(i))

    str_vector_sign_layer_3 = ''
    for i in vector_sign_layer_3:
        str_vector_sign_layer_3 += str(int(i))

    print(str_vector_sign_layer_3)
    sram_addr = 0
    with open('lenet/sram_sign_hex_stream_1.txt', 'w+') as f:
        for k in range(int(len(str_vector_sign_layer_1)/8)):
            f.write("init_sign[{}] = 8'b{};\n".format(sram_addr, str_vector_sign_layer_1[8*k: 8*k+8]))
            sram_addr += 1

    with open('lenet/sram_sign_hex_stream_2.txt', 'w+') as f:
        for k in range(int(len(str_vector_sign_layer_2)/8)):
            f.write("init_sign[{}] = 8'b{};\n".format(sram_addr, str_vector_sign_layer_2[8*k: 8*k+8]))
            sram_addr += 1

    with open('lenet/sram_sign_hex_stream_3.txt', 'w+') as f:
        for k in range(int(len(str_vector_sign_layer_3)/8)):
            f.write("init_sign[{}] = 8'b{};\n".format(sram_addr, str_vector_sign_layer_3[8*k: 8*k+8]))
            sram_addr += 1

    # sram_addr = 0
    # with open('lenet/positive_hex_stream_1.txt', 'w') as f:
    #     f.write(positive_hex_stream_1)
    #     f.write("\n" + str(len(positive_hex_stream_1)*4))
    # with open('lenet/sram_positive_hex_stream_1.txt', 'w+') as f:
    #     for k in range(int(len(positive_hex_stream_1)/64)):
    #         f.write("init_data[{}] = 256'h{};\n".format(sram_addr, positive_hex_stream_1[64*k: 64*k+64]))
    #         sram_addr += 1
    
    # with open('lenet/negative_hex_stream_1.txt', 'w') as f:
    #     f.write(negative_hex_stream_1)
    #     f.write("\n" + str(len(negative_hex_stream_1)*4))
    # with open('lenet/sram_negative_hex_stream_1.txt', 'w+') as f:
    #     for k in range(int(len(negative_hex_stream_1)/64)):
    #         f.write("init_data[{}] = 256'h{};\n".format(sram_addr, negative_hex_stream_1[64*k: 64*k+64]))
    #         sram_addr += 1

    # with open('lenet/positive_hex_stream_2.txt', 'w') as f:
    #     f.write(positive_hex_stream_2)
    #     f.write("\n" + str(len(positive_hex_stream_2)*4))
    # with open('lenet/sram_positive_hex_stream_2.txt', 'w+') as f:
    #     for k in range(int(len(positive_hex_stream_2)/64)):
    #         f.write("init_data[{}] = 256'h{};\n".format(sram_addr, positive_hex_stream_2[64*k: 64*k+64]))
    #         sram_addr += 1
   
    # with open('lenet/negative_hex_stream_2.txt', 'w') as f:
    #     f.write(negative_hex_stream_2)
    #     f.write("\n" + str(len(negative_hex_stream_2)*4))
    # with open('lenet/sram_negative_hex_stream_2.txt', 'w+') as f:
    #     for k in range(int(len(negative_hex_stream_2)/64)):
    #         f.write("init_data[{}] = 256'h{};\n".format(sram_addr, negative_hex_stream_2[64*k: 64*k+64]))
    #         sram_addr += 1

    # with open('lenet/positive_hex_stream_3.txt', 'w') as f:
    #     f.write(positive_hex_stream_3)
    #     f.write("\n" + str(len(positive_hex_stream_3)*4))
    # with open('lenet/sram_positive_hex_stream_3.txt', 'w+') as f:
    #     for k in range(int(len(positive_hex_stream_3)/64)):
    #         f.write("init_data[{}] = 256'h{};\n".format(sram_addr, positive_hex_stream_3[64*k: 64*k+64]))
    #         sram_addr += 1

    # with open('lenet/negative_hex_stream_3.txt', 'w') as f:
    #     f.write(negative_hex_stream_3)
    #     f.write("\n" + str(len(negative_hex_stream_3)*4))
    # with open('lenet/sram_negative_hex_stream_3.txt', 'w+') as f:
    #     for k in range(int(len(negative_hex_stream_3)/64)):
    #         f.write("init_data[{}] = 256'h{};\n".format(sram_addr, negative_hex_stream_3[64*k: 64*k+64]))
    #         sram_addr += 1

    # ## write bias
    # with open('lenet/bias_hex_stream_1.txt', 'w') as f:
    #     f.write(bias_hex_stream_1)
    #     f.write("\n" + str(len(bias_hex_stream_1)*4))
    # with open('lenet/sram_bias_hex_stream_1.txt', 'w+') as f:
    #     for k in range(int(len(bias_hex_stream_1)/64)):
    #         f.write("init_data[{}] = 256'h{};\n".format(sram_addr, bias_hex_stream_1[64*k: 64*k+64]))
    #         sram_addr += 1

    # with open('lenet/bias_hex_stream_2.txt', 'w') as f:
    #     f.write(bias_hex_stream_2)
    #     f.write("\n" + str(len(bias_hex_stream_2)*4))
    # with open('lenet/sram_bias_hex_stream_2.txt', 'w+') as f:
    #     for k in range(int(len(bias_hex_stream_2)/64)):
    #         f.write("init_data[{}] = 256'h{};\n".format(sram_addr, bias_hex_stream_2[64*k: 64*k+64]))
    #         sram_addr += 1

    # with open('lenet/bias_hex_stream_3.txt', 'w') as f:
    #     f.write(bias_hex_stream_3)
    #     f.write("\n" + str(len(bias_hex_stream_3)*4))
    # with open('lenet/sram_bias_hex_stream_3.txt', 'w+') as f:
    #     for k in range(int(len(bias_hex_stream_3)/64)):
    #         f.write("init_data[{}] = 256'h{};\n".format(sram_addr, bias_hex_stream_3[64*k: 64*k+64]))
    #         sram_addr += 1

    # get hex streams from int
    # positive_indices_hex_list_1, positive_indices_hex_stream_1 = Dec2Hex(positive_integrating_lengths_1)
    # positive_indices_hex_list_2, positive_indices_hex_stream_2 = Dec2Hex(positive_integrating_lengths_2)
    # positive_indices_hex_list_3, positive_indices_hex_stream_3 = Dec2Hex(positive_integrating_lengths_3)

    # negative_indices_hex_list_1, negative_indices_hex_stream_1 = Dec2Hex(negative_integrating_lengths_1)
    # negative_indices_hex_list_2, negative_indices_hex_stream_2 = Dec2Hex(negative_integrating_lengths_2)
    # negative_indices_hex_list_3, negative_indices_hex_stream_3 = Dec2Hex(negative_integrating_lengths_3)

    # with open('lenet/positive_indices_hex_stream_1.txt', 'w') as f:
    #     f.write(positive_indices_hex_stream_1)
    #     f.write("\n" + str(len(positive_indices_hex_stream_1)*4))
    
    # with open('lenet/negative_indices_hex_stream_1.txt', 'w') as f:
    #     f.write(negative_indices_hex_stream_1)
    #     f.write("\n" + str(len(negative_indices_hex_stream_1)*4))

    # with open('lenet/positive_indices_hex_stream_2.txt', 'w') as f:
    #     f.write(positive_indices_hex_stream_2)
    #     f.write("\n" + str(len(positive_indices_hex_stream_2)*4))
    
    # with open('lenet/negative_indices_hex_stream_2.txt', 'w') as f:
    #     f.write(negative_indices_hex_stream_2)
    #     f.write("\n" + str(len(negative_indices_hex_stream_2)*4))
    
    # with open('lenet/positive_indices_hex_stream_3.txt', 'w') as f:
    #     f.write(positive_indices_hex_stream_3)
    #     f.write("\n" + str(len(positive_indices_hex_stream_3)*4))
    
    # with open('lenet/negative_indices_hex_stream_3.txt', 'w') as f:
    #     f.write(negative_indices_hex_stream_3)
    #     f.write("\n" + str(len(negative_indices_hex_stream_3)*4))