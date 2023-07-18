from socket import *
import argparse


def SendEthPacket(src, dst, eth_type, payload, interface, verbose=False):
  assert(len(src) == len(dst) == 6) # 6 bytes MAC addresses
  assert(len(eth_type) == 2) # 2 bytes ethernet type
  # assert(len(payload) == 50) # 50 bytes analog data stream, where each analog data is 16 bits (2 bytes)

  s = socket(AF_PACKET, SOCK_RAW)
  s.bind((interface, 0))    
  padded_payload = bytes.fromhex('0' * 4)
  s.send(dst + src + eth_type + padded_payload + payload)

  if verbose: 
    print("payload", payload)
    print("padded_payload", padded_payload)
    print("SendEthPacket done")


def ConstructEthPacket(data_to_send):
  payloads_to_send = []
  return payloads_to_send


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--payload", help="payload data of a packet", type=str)
  args = parser.parse_args().payload
  print(args)

  src = b'\x04\x3F\x72\xF4\x3E\xD6'
  dst = b'\xAA\xBB\xCC\xDD\xEE\xFF'
  eth_type = b'\x7A\x05'
  interface = "ens4f1"  # 100G NIC on the cambridge server 

  ## a sine wave, total 50 bytes: the first 18 bytes are idle, then 32 bytes (each hex is 16 bit dec) carry data
  sinewave = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xCF\x07\xA5\x7F\x89\xC3\x80\x03\x89\xC3\xA5\x7F\xCF\x07\xFF\xFF\x30\xF8\x5A\x80\x76\x3C\x7F\xFC\x76\x3C\x5A\x80\x30\xF8\x00\x00\xCF\x07\xA5\x7F\x89\xC3\x80\x03\x89\xC3\xA5\x7F\xCF\x07\xFF\xFF\x30\xF8\x5A\x80\x76\x3C\x7F\xFC\x76\x3C\x5A\x80\x30\xF8\x00\x00'  # a sine symbol
  two_sine_stream = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xA5\x7F\x80\x03\xA5\x7F\xFF\xFF\x5A\x80\x7F\xFC\x5A\x80\x00\x00\xA5\x7F\x80\x03\xA5\x7F\xFF\xFF\x5A\x80\x7F\xFC\x5A\x80\x00\x00'
  square_wave = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80\x03\x00\x00\x80\x03\x00\x00\x80\x03\x00\x00\x80\x03\x00\x00\x80\x03\x00\x00\x80\x03\x00\x00\x80\x03\x00\x00\x80\x03\x00\x00'
  ## send packets
  for i in range(10000):
    SendEthPacket(src, dst, eth_type, square_wave, interface)
    print(i, "packet sent")
    # time.sleep(10)
  
