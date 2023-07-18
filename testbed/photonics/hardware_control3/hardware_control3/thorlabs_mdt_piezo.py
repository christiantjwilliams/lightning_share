import PyThorlabsMDT

class ThorlabsPiezo(object):
	def __init__(self, port):
		self.pzt = PyThorlabsMDT.PZT_driver(port)

	def info(self):
		return self.pzt.get_info()

	def get_voltages(self):
		return self.pzt.x, self.pzt.y, self.pzt.z

	def set_voltages(self, vx, vy, vz):
		if vx is not None:
			vx = round(float(vx), 2)
		if vy is not None:
			vy = round(float(vy), 2)
		if vz is not None:
			vz = round(float(vz), 2)

		if (vx > 75.0) or (vy > 75.0) or (vz > 75.0):
			print('Invalid voltage. Must be less than 75.0 Volts.')
			return 0
		else:
			if vx is not None: self.pzt.x = vx
			if vy is not None: self.pzt.y = vy
			if vz is not None: self.pzt.z = vz

if __name__ == '__main__':
	x = ThorlabsPiezo('/dev/tty.usbmodem1472')
	print(x.info())
	#print x.get_voltages()
