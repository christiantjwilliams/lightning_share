import pickle
import numpy as np
from lenet_weight_data import *

    
if __name__ == "__main__":
    # MNIST LeNet data
    mnist_data = pickle.load(open("/home/zhizhenz/lightning-master/data/saved_activation/lenet/mnistdata.p", "rb"))

    # image_list = [np.array(mnist_data.dataset.data[i,:,:]) for i in range(mnist_data.dataset.data.shape[0])]
    
    # for now let us only consider the first 100 pictures
    image_list = [np.array(mnist_data.dataset.data[i,:,:]) for i in range(100)]
    rescale_multiple_images = RescaleData(image_list)

    for i in range(100):
        locals()["positive_layer_"+str(i)], locals()["negative_layer_"+str(i)] = SeparateNegPos(rescale_multiple_images[i])

    for i in range(100):
        locals()["converted_positive_data_"+str(i)], locals()["positive_indices_"+str(i)], locals()["positive_integrating_lengths_"+str(i)] = PreserveDataConvert(locals()["positive_layer_"+str(i)], "positive")
        # locals()["converted_negative_data_"+str(i)], locals()["negative_indices_"+str(i)], locals()["negative_integrating_lengths_"+str(i)] = PreserveDataConvert(locals()["negative_layer_"+str(i)], "negative")
    
    for i in range(100):
        locals()["positive_hex_list_"+str(i)], locals()["positive_hex_stream_"+str(i)] = Dec2Hex(locals()["converted_positive_data_"+str(i)])
        # locals()["negative_hex_list_"+str(i)], locals()["negative_hex_stream_"+str(i)] = Dec2Hex(locals()["converted_negative_data_"+str(i)])

    sram_addr = 0
    for i in range(100):
        with open('lenet/sram_activation_positive_hex_stream_{}.txt'.format(i), 'w+') as pf:
            for k in range(int(len(locals()["positive_hex_stream_"+str(i)])/64)):
                pf.write("init_data[{}] = 256'h{};\n".format(sram_addr, locals()["positive_hex_stream_"+str(i)][64*k: 64*k+64]))
                sram_addr += 1
            pf.close()

        # with open('lenet/sram_activation_negative_hex_stream_{}.txt'.format(i), 'w+') as nf:
        #     for k in range(int(len(locals()["negative_hex_stream_"+str(i)])/64)):
        #         nf.write("init_data[{}] = 256'h{};\n".format(sram_addr, locals()["negative_hex_stream_"+str(i)][64*k: 64*k+64]))
        #         sram_addr += 1
        #     nf.close()