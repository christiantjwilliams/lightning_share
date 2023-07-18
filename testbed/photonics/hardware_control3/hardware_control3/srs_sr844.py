import visa
import time
import numpy as np

class srs_sr844(object):

	"""
	Visa functions for SRS SR844 lock-in amplifier
	"""
	def __init__(self, address='GPIB0::8'):
		rm = visa.ResourceManager()
		self.instr = rm.open_resource(address)
		self.name = self.instr.query('*IDN?')
		self.__get_internal_ref_mode()

	def reset(self):
		self.instr.write('*RST')

	def __get_internal_ref_mode(self):
		val = int(self.instr.query('FMOD?'))
		if val is 1:
			self.ref_mode = "internal"
		elif val is 0:
			self.ref_mode = "external"
		pass

	def get_internal_ref_mode(self):
		self.__get_internal_ref_mode()
		return self.ref_mode

	def set_internal_ref_mode(self, internal=True):
		if internal:
			self.instr.write('FMOD 1')
		else: # set to external
			self.instr.write('FMOD 0')
		self.__get_internal_ref_mode()
		pass	

	def set_ref_freq(self, freq):
		if self.ref_mode is not "internal":
			raise NotImplementedError("not using internal frequency generator")
		self.instr.write('FREQ '+str(freq)) # in Hz
		pass

	def get_ref_freq(self):
		return float(self.instr.query('FREQ?'))

	# AUX voltage inputs and outputs
	def set_aux_output_voltage(self, volt, chan=1):
		self.instr.write('AUXO '+str(chan)+','+str(volt))
		pass

	def get_aux_output_voltage(self, chan=1):
		return float(self.instr.query('AUXO?'+str(chan)))

	def get_aux_input_voltage(self, chan=1):
		return float(self.instr.query('AUXI '+str(chan)))

	# Measurement commands
	def __parse_string_to_vector(self, string):
		s = np.array(string.split(','))
		return s.astype(np.float)

	def get_XY(self): # get X,Y position of the signal
		output_string = self.instr.query('snap? 1,2')
		return self.__parse_string_to_vector(output_string)

	def get_RT(self): # get R,Theta position of the signal
		output_string = self.instr.query('snap? 3,5')
		return self.__parse_string_to_vector(output_string)

	def get_XYRT(self): # get X,Y,R,Theta
		output_string = self.instr.query('snap? 1,2,3,5')
		return self.__parse_string_to_vector(output_string)

	def get_ch1ch2(self): # get CH1 and CH2 displays
		output_string = self.instr.query('snap? 9,10')
		return self.__parse_string_to_vector(output_string)
