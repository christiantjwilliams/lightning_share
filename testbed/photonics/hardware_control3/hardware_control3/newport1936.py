import os
import ctypes
import numpy as np
import time

READ_BUFFER_SIZE=64
DLL_NAME=os.path.join(os.path.dirname(os.path.realpath(__file__)),"usbdll.dll")
SEP_STRING="\r\n"
MAX_RANGE=7

class Newport1936(object):
	"""Python class for Newport 1936R power meter.
	The device does not use GPIB, so USB connection is needed.
	Works only on Windows because of the WinDLL"""

	def __init__(self, pid=0xcec7):
		"""pid here is the product ID found in NewportPwrMtr.inf,
		the value doesn't change unless you're using old Windows"""
		self.pid = pid
		self.connection = USBConnection(pid)

		self.setup_basic()

	def _build_range_dictionary(self): # will be in dBm
		self.rangeDic={}
		for r in range(MAX_RANGE+1):
			self.set_range(auto=False,level=r)
			self.rangeDic[r]=self.get_max_power()
		self.set_range(auto=True)

	def range_dictionary(self):
		return self.rangeDic

	def reset(self):
		del self.connection
		self.connection = USBConnection(self.pid)

	def setup_basic(self):
		# disable echoing commands back (slow, useful only for debugging)
		self.connection.write('ECHO SET 0')
		# measure in dBm
		self.set_units(units='dBm')
		# build range dictionary
		# self._build_range_dictionary()
		# set to automatic ranging
		self.set_range(auto=True)
		# set wavelength to 1550 nm
		self.set_wavelength(1550)
		# set to channel 1
		self.set_channel(1)

	def set_units(self, units='dBm'):
		if units is 'dBm':
			self.connection.write('UNITS 6')
		elif units is 'Watts':
			self.connection.write('UNTIS 2')
		else:
			raise ValueError("unknown units")

	def read_power(self):
		return self.connection.readFloat('PM:P?')

	def set_channel(self,chan=1):
		self.connection.write('PM:CHAN %i' % chan)

	def get_channel(self):
		return self.connection.readFloat('PM:CHAN?')

	def get_max_power(self):
		""" returns the maximum readable power for the current float """
		remeasure=1
		while remeasure < 5:
			try:
				return self.connection.readFloat("PM:MAX:Power?")
			except Exception as e:
				remeasure+=1
		raise e

	def set_wavelength(self,wav): # in nm
		self.connection.write('PM:LAMBDA %.0f' % wav) # 2 decimal places

	def get_wavelength(self): # in nm
		return self.connection.readFloat('PM:LAMBDA?')

	def set_range(self,auto=True,level='7'):
		if auto:
			self.connection.write('PM:AUTO 1')
		else: # Range goes from 0 to 7, with 0 being the highest gain
			self.connection.write('PM:AUTO 0')
			if (level < 0) or (level > 7):
				raise ValueError("invalid level, can only go from 0-7")
			self.connection.write('PM:RANGE %i' % level)

class USBConnection(object):
	""" Abstraction of the low level connection to USB bus so that destructor can be used without circular
	references as per http://eli.thegreenplace.net/2009/06/12/safely-using-destructors-in-python/.
	This class is essentially a wrapper for the Newport USB Driver library usbdll.dll"""
	def __init__(self,pid):
		""" Open the USB connection and get the device ID for future communication """
		self.readBufferSize=READ_BUFFER_SIZE
		try:
			self.lib=ctypes.WinDLL(DLL_NAME)
		except Exception as e:
			raise PowerMeterLibraryError("Could not load the power meter library " + DLL_NAME + ". \n" + e.args[0])
		# Open the usb device specified by pid
		s=self.lib.newp_usb_open_devices(pid,False,ctypes.byref(ctypes.c_int(0)))
		if s<0:
			raise CommError("Connection to pid=" + str(pid) + " could not be initialized and returned " + str(s))
		# Retrieve the device ID assigned above
		readBuffer=ctypes.create_string_buffer(1024)
		s=self.lib.newp_usb_get_device_info(readBuffer)
		if s<0:
			raise CommError("Connection to pid=" + str(pid) + " successful, but get_device_info failed and returned " + str(s))
		deviceInfoList=readBuffer.value.split(',')
		if len(deviceInfoList)>2:
			raise CommError("More than one Newport instrument detected on the USB bus which is not supported")
		self.id=int(deviceInfoList[0])
	def __del__(self):
		""" Destructor method unitializes the USB connection """
		self.lib.newp_usb_uninit_system()
	def write(self,writeString):
		""" Writes a single command to the USB device"""
		s=self.lib.newp_usb_send_ascii(self.id, writeString+SEP_STRING, len(writeString+SEP_STRING))
		if s<0:
			raise CommError("Writing of command '" + writeString + "' was not succesful and returned " + str(s))
	
	def read(self):
		""" Reads the response from power meter """
		readBuffer=ctypes.create_string_buffer(self.readBufferSize)
		numBytesRead=ctypes.c_int(0)
		s=self.lib.newp_usb_get_ascii(self.id,readBuffer,self.readBufferSize,ctypes.byref(numBytesRead))
		if s<0:
			raise CommError("Reading from power meter was not succesful and returned " + str(s))
		return readBuffer.value

	   
	def readFloat(self,queryString):
		""" Writes the query command in queryString, then reads the response and returns it """
		self.write(queryString)
		returnString=self.read()
		try:
			return float(returnString.strip())
		except ValueError:
			self.clearComQueue()
			self.write(queryString)
			returnString=self.read()
			return float(returnString.strip())


	def clearComQueue(self):
		""" Clears the communication queue in case communication was interrupted somehow. Probably not needed. """
		commStr=""
		while 1:
			try:
				commStr+=self.read()
			except CommError:
				break
		return commStr

# Error classes
class CommError(Exception): pass
class PowerMeterLibraryError(Exception): pass


# For minimal working testing purposes
if __name__ == '__main__':

	pm = Newport1936()
	# test by measuring current power
	time.sleep(2)
	print(pm.read_power())

	# test wavelength settings
	pm.set_wavelength(1565)
	print(pm.get_wavelength())

	# # test range dictionaries
	# rangeDic = pm.range_dictionary()
	# print rangeDic
