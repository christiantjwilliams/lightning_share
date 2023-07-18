from packetgen import SendEthPacket
import pyshark
import concurrent.futures
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from multiprocessing import Process


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

    return hex_list, hex_stream


## input a list of hex (16 bits) strings, output a list of decimal integers
def Hex2Dec(hex_streams, reverse, plotsign):
    hex_stream_set = hex_streams.split('_')
    # print("hex_stream_set", hex_stream_set)
    bit16_dec_stream_set = []
    bit14_dec_stream_set = []

    if reverse:
        for i in range(len(hex_stream_set)):
            hex_value = '0x'+ hex_stream_set[len(hex_stream_set)-1-i]
            value16 = int(hex_value, 16)
            bit16_dec_stream_set.append(-(value16 & 0x8000) | (value16 & 0x7fff))  # two's complement signed integer using bitwise operations
            hex_value = hex(-(value16 & 0x8000) >> 2 | (value16 & 0x7ffc) >> 2)
            value14 = int(hex_value, 16)
            bit14_dec_stream_set.append(-(value14 & 0x8000) | (value14 & 0x7ffc))  # two's complement signed integer using bitwise operations
    else:
        for i in range(len(hex_stream_set)):
            hex_value = '0x'+ hex_stream_set[i]
            value16 = int(hex_value, 16)
            bit16_dec_stream_set.append(-(value16 & 0x8000) | (value16 & 0x7fff))  # two's complement signed integer using bitwise operations
            hex_value = hex(-(value16 & 0x8000) >> 2 | (value16 & 0x7ffc) >> 2)
            value14 = int(hex_value, 16)
            bit14_dec_stream_set.append(-(value14 & 0x8000) | (value14 & 0x7ffc))  # two's complement signed integer using bitwise operations

    # print("bit16_dec_stream_set", bit16_dec_stream_set)
    # print("bit14_dec_stream_set", bit14_dec_stream_set)

    if plotsign:
        plt.figure()
        plt.plot(bit16_dec_stream_set, marker=".", label="16 bits (AXI stream)")
        plt.plot(bit14_dec_stream_set, marker=".", label="14 bits (FPGA output)")
        plt.title("DAC analog output")
        plt.legend(loc="best")
        plt.show()

    return bit14_dec_stream_set


def PacketGenertor(payload, verbose=False):
    # print(" - Generating a packet")
    src = b'\x04\x3F\x72\xF4\x3E\xD6'  # MAC addr
    dst = b'\xAA\xBB\xCC\xDD\xEE\xFF'  # MAC addr
    eth_type = b'\x7A\x05'
    interface = "ens4f1"  # 100G NIC on the cambridge server
    SendEthPacket(src, dst, eth_type, bytes.fromhex(payload), interface, verbose=verbose)
    # print("Packet sent")


def PacketCapture(interface, capture_pkt_num):
    print(" - Capturing packets at interface", interface)
    capture = pyshark.LiveCapture(interface)
    # capture.sniff(packet_count=capture_pkt_num)
    capture.sniff(timeout=10)
    print("{} packet captured!".format(len(capture)))
    return capture


# tvalid of ethernet transmit is config_regs[0][0]
def FlipConfigReg(reg_index, addr, reg_value_1, reg_value_2, seconds):
    # 'PATH', 'ADDR', 'DATA'
    os.system("python ../../fpga/lib/xfcp/python/xfcp_ctrl.py --enum")
    os.system("python ../../fpga/lib/xfcp/python/xfcp_ctrl.py --write {} {} {}".format(reg_index, addr, reg_value_1))
    os.system("python ../../fpga/lib/xfcp/python/xfcp_ctrl.py --read {} {} 2".format(reg_index, addr))
    time.sleep(seconds)
    # 'PATH', 'ADDR', 'DATA'
    os.system("python ../../fpga/lib/xfcp/python/xfcp_ctrl.py --write {} {} {}".format(reg_index, addr, reg_value_2))
    os.system("python ../../fpga/lib/xfcp/python/xfcp_ctrl.py --read {} {} 2".format(reg_index, addr))



def RepetitiveSend(hex_stream):
    chunk = 2400
    repetiton = len(hex_stream) // chunk  # 2B dpacket
    for i in range(repetiton):
        PacketGenertor(hex_stream[chunk*i : chunk*(i+1)])


def Send2ONN(pkt_data_tosend):
    hex_list, hex_stream = Dec2Hex(pkt_data_tosend)

    ## send packets
    per_process_data_volumn = len(hex_stream)//32
    processes = [Process(target=RepetitiveSend, args=(hex_stream[per_process_data_volumn*p : per_process_data_volumn*(p+1)],)) for p in tqdm(range(32))]
    for i in range(32):
        processes[i].start()

## edit a certain config register
def EditConfigRegs(reg_index, bit_info):
    hex_reg_index = hex(reg_index*2)
    os.system("python ../../fpga/lib/xfcp/python/xfcp_ctrl.py --write 0 {} {}".format(hex_reg_index, bit_info))  


def EthernetRead(capture_pkt_num, layer, propagation_delay):
    capture = []
    interface = "ens4f1"
    ethernet_receive_values = []

    direct_hex_delay = hex(propagation_delay)[2:]
    if len(direct_hex_delay) < 2:
        hex_delay = '0' * (2-len(direct_hex_delay)) + direct_hex_delay
        hex_delay += '00'
    else:
        hex_delay = direct_hex_delay[-2:] + '0' * (4-len(direct_hex_delay)) + direct_hex_delay[:-2]
   
    EditConfigRegs(2, hex_delay)

    if layer == "first_positive":
        code = "0900"
    elif layer == "first_negative":
        code = "0100"
    elif layer == "second_positive":
        code = "1200"
    elif layer == "second_negative":
        code = "0200"
    elif layer == "third_positive":
        code = "2400"
    elif layer == "third_negative":
        code = "0400"

    with concurrent.futures.ThreadPoolExecutor() as executor:
        packet_capture = executor.submit(
            PacketCapture,
            interface,
            capture_pkt_num,
        )
        time.sleep(1)
        flip_config_reg = executor.submit(
            EditConfigRegs,
            1, 
            code,  
        )
        capture = packet_capture.result()
    
    print("processing captured packets...")
    for i in tqdm(range(len(capture))):
        for h in range(int(len(capture[i].data.data[4:36])/4)):
            hex_data = capture[i].data.data[4+h*4+2: 4+h*4+4] + capture[i].data.data[4+h*4: 4+h*4+2]
            num = Hex2Dec(hex_data, False, False)
            if len(num):
                ethernet_receive_values.append(num[0])
    
    return ethernet_receive_values


if __name__ == "__main__":
    # pkt_data_tosend = range(-8192, 8191, 1)
    pkt_data_tosend = []
    
    temp = np.linspace(-8192,8191,num=100)
    temp = [int(i) for i in temp]
    for i in range(1000000):
        # pkt_data_tosend = [0, 2000, 4000, 6000, 8000, 6000, 4000, 2000, 0, -2000, -4000, -6000, -8000, -6000, -4000]
        pkt_data_tosend += temp
        
    # print(pkt_data_tosend)
    ## sending data to DAC ports
    print("start sending")
    Send2ONN(pkt_data_tosend)
    print("finished..")

    ## receiving data from ADC ports
    # captured_packet_data = Read2ONN(capture_pkt_num=len(pkt_data_tosend)//8)

    ## parallel sending and receiving packets
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     packet_sent = executor.submit(
    #         Send2ONN,
    #         pkt_data_tosend,
    #     )
    #     packet_captured = executor.submit(
    #         Read2ONN,
    #         len(pkt_data_tosend)//8,
    #     )

    #     captured_packet_data = packet_captured.result()

    # print(captured_packet_data)
