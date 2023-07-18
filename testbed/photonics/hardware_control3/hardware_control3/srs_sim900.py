import serial
import inspect
import numpy as np
import time

# SIM 900 Global variables
TIMEOUT = .1
DEFAULT_BAUD = 9600
ESCAPE_KEY = "XYZZY"

# SIM 928 Global variables
MAX_VOLTAGE = 5 # in V

class srs_sim928(object):
	"""
	Class for SIM928 methods. Outputs the serial command needed for SIM900.
	"""
	def __init__(self):
		pass

	def __check_voltage(self, v):
		if abs(v) > MAX_VOLTAGE:
			raise ValueError
		pass

	def set_voltage(self, v):
		self.__check_voltage(v)
		return "VOLT %.3f\n" %v, None

	def get_voltage(self):
		return "VOLT?\n", float

	def set_output(self, output):
		if output:
			return "OPON\n", None
		else:
			return "OPOF\n", None

	def get_output(self):
		return "EXON?\n", str


class srs_sim900(object):
	"""
	Pass the instruments when initializing. See example below.
	"""
	def __init__(self, port, instruments):
		# Check if the keys are correct
		for k in instruments.keys():
			assert isinstance(k, int) and (k>0) and (k<9), \
				"Slot must be an integer between 1-9"
		self.instruments = instruments

		# Create serial connection with main frame	
		try:
			self.main_frame = \
				serial.Serial(port, timeout=TIMEOUT, baudrate=DEFAULT_BAUD)
		except serial.SerialException:
			print("Cannot connect to SIM900 main frame")
		else:
			self.main_frame.write('*IDN?\n')
			self.name = str(self.main_frame.readline().strip())
		pass

	def execute(self, slot, func, **kwargs):
		try:
			inst = self.instruments[slot]
		except KeyError:
			print("No such slot")
			return
		
		# Connect to the specific slot
		self.main_frame.write("conn "+str(slot)+", \'"+ESCAPE_KEY+"\'\n")

		value = None
		try:
			comm, val_type = \
				func(*tuple(value for _, value in kwargs.items()))
			self.main_frame.write(comm)
			if val_type is not None: # means we're getting a value back
				value = list(map(val_type, self.main_frame.readline().split()))
		except AttributeError:
			print("Wrong function, check your instrument and method")
		except (ValueError, TypeError):
			print("Problems with command")
		finally: 
			self.main_frame.write(ESCAPE_KEY) # disconnect back to main frame
			if value is not None:
				return value

	def __del__(self):
		self.main_frame.close()

if __name__ == '__main__':
	# We have SRS SIM 928 modules connected in ports 5,6,7,8
	s = srs_sim928()
	instruments = {5:s, 6:s, 7:s, 8:s}
	sim900 = srs_sim900('COM9', instruments)

	v_list = np.linspace(0,1.0,11)
	sim900.execute(7, s.set_output, output=True)
	for v in v_list:
		sim900.execute(7, s.set_voltage, v=v)
		time.sleep(.1)
	
	time.sleep(2)
	sim900.execute(7, s.set_output, output=False)

	input('Done with commands, press enter to continue...')
