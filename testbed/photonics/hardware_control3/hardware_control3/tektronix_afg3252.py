import visa
import numpy as np
import ctypes
from struct import pack
import matplotlib.pyplot as plt

FREQ_UNITS = ["Hz", "kHz", "MHz"]
VOLTAGE_UNITS = ["mV", "V"]
VOLTAGE_TYPES = ["VPP", "VRMS", "DBM"]
PHASE_UNITS = ["RAD", "DEG"]
TIME_UNITS = ["ns", "us", "ms", "s"]
WAVEFORM_TYPES = ["SIN", "SQUARE", "PULSE", "RAMP", "SINC", "GAUSSIAN", "LORENTZ",\
									"USER1", "USER2", "USER3", "USER4"]
HISPEED_MAX_RANGE = 16384
HISPEED_MIN_RANGE = 2
LOSPEED_MAX_RANGE = 131072
LOSPEED_MIN_RANGE = 16385
DATA_MAX_RANGE = 16382
DATA_MIN_RANGE = 0

class TektronixAFG3252(object):
	"""
	Python wrapper for Tektronix AFG3252C dual-channel arbitrary waveform 
	generator.
	"""
	def __init__(self, address="GPIB0::11", timeout=10000):
		rm = visa.ResourceManager()
		self.instr = rm.open_resource(address)

		self.name = self.instr.query('*IDN?')
		self.__timeout = timeout # in ms
		self.instr.timeout = self.__timeout

	def _check_freq_units(self,units):
		if units not in FREQ_UNITS:
			raise ValueError("Unknown frequency unit")

	def _check_voltage_units(self,units):
		if units not in VOLTAGE_UNITS:
			raise ValueError("Unknown voltage unit")
	
	def _check_voltage_types(self,units):
		if units not in VOLTAGE_TYPES:
			raise ValueError("Unknown voltage type")

	def _check_phase_units(self,units):
		if units not in PHASE_UNITS:
			raise ValueError("Unknown phase unit")

	def _check_time_units(self,units):
		if units not in TIME_UNITS:
			raise ValueError("Unknown time unit")

	def _check_waveform_types(self,waveform):
		if waveform not in WAVEFORM_TYPES:
			raise ValueError("Unknown waveform")

	def reset(self):
		self.instr.write('*RST')
		self.instr.timeout = self.__timeout

	def set_waveform(self, waveform, chan=1):
		self._check_waveform_types(waveform)
		self.instr.write('SOURCE%i:FUNCTION:SHAPE %s' % (chan, waveform))

	def set_frequency(self, freq, units="Hz", chan=1):
		self._check_freq_units(units)
		self.instr.write('SOURCE%i:FREQUENCY %f %s' % (chan, freq, units))

	def set_voltage_amplitude(self, voltage, units="VPP", chan=1):
		# precision is at 0.1 mVpp (4 digits)
		self._check_voltage_types(units)
		self.instr.write('SOURCE%i:VOLTAGE:LEVEL %f%s' % (chan, voltage, units))

	def set_voltage_offset(self, voltage, units="V", chan=1):
		# precision is at 0.1 mV (4 digits)
		self._check_voltage_units(units)
		self.instr.write('SOURCE%i:VOLTAGE:OFFSET %f%s' % (chan, voltage, units))

	def set_phase(self, phase, units="DEG", chan=1):
		# only applies when the waveform is not DC, noise or pulse
		self._check_phase_units(units)
		self.instr.write('SOURCE%i:PHASE:ADJUST %f%s' % (chan, phase, units))

	def set_pulse_dutycycle(self, dcycle, chan=1):
		self.instr.write('SOURCE%i:PULSE:DCYCLE %.4f' % (chan, dcycle))

	def set_pulse_period(self, period, units="ms", chan=1):
		self._check_time_units(units)
		self.instr.write('SOURCE%i:PULSE:PERIOD %f %s' % (chan, period, units))

	def set_pulse_delay(self, delay, units="ms", chan=1):
		self._check_time_units(units)
		self.instr.write('SOURCE%i:PULSE:DELAY %f%s' %(chan, delay, units))

	def set_pulse_width(self, width, units="ms", chan=1):
		self._check_time_units(units)
		self.instr.write('SOURCE%i:PULSE:WIDTH %f%s' %(chan, width, units))

	def set_output(self, output=True, chan=1):
		if output:
			self.instr.write('OUTPUT%i:STATE ON')
		else:
			self.instr.write('OUTPUT%i:STATE OFF')


	def insert_arb_waveform(self, x, save_chan=1):
		# check if the number of points in the waveform exceeds what's allowed
		N = len(x)
		if (N < HISPEED_MIN_RANGE) or (N > LOSPEED_MAX_RANGE):
			raise ValueError("Invalid number of points in the arbitrary waveform")
		# check that the range of x is from 0 to 1
		if (np.min(x) < 0) or (np.max(x) > 1) :
			raise ValueError("x must be bounded between 0 and 1")

		intData = x * DATA_MAX_RANGE

		# Normalize data to between 0 and 16,382
		minData = intData.min()
		maxData = intData.max()
		intData = (intData-minData) * (DATA_MAX_RANGE/(maxData-minData))
		intData = intData.astype('>u2')

		bytesData = intData.tostring()
		num = len(bytesData)
		bytesData = b"#{}{}".format(len(str(num)), num) + bytesData
		self.instr.write_raw(b'DATA:DATA EMEMory,' + bytesData)

		# save waveform in one of the preset memories
		self.instr.write("DATA:COPY USER%i,EMEM" % save_chan)

if __name__ == '__main__':
	obj = TektronixAFG3252()
	obj.reset()
	
	# create a triangle
	# N = 100
	# triangle = np.empty(N)
	# for i in range(N/2):
	# 	triangle[i] = i / float(N/2)
	# for i in range(N/2, N):
	# 	triangle[i] = (N/2-float(i-N/2)) / float(N/2)

	# obj.insert_arbitrary_waveform(triangle, save_chan=1)

	# sinusoid
	N = 200
	sinusoid = np.empty(N)
	for i in range(N):
		sinusoid[i] = .5*np.sin(float(i*20)/N) + .5

	obj.insert_arb_waveform(sinusoid, save_chan=4)

	plt.plot(sinusoid)
	plt.show()
