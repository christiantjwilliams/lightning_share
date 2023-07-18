import visa
import time
import numpy as np


class EXFOFVA3100(object):

    def __init__(self, address):
        rm = visa.ResourceManager()
        self.instr = rm.open_resource(address)
        self.name = self.instr.query('*IDN?')
        self.delay = 0.1

    def setup_basic(self):
    	self.instr.write('CLR')
        self.instr.write('ATT 0 DB')
        self.multimode_fiber(False)

    def set_attenuation(self, n):
        """
        Attenuation in decibels (DB). Must be negative floating point.
        """
        time.sleep(self.delay)
        att = np.round(float(n)/0.005)*0.005
        self.instr.write('ATT ' + str(np.around(att,3)) + ' DB')

    def get_attenuation(self):
    	time.sleep(self.delay)
        return self.instr.query('ATT?')

    def set_wavelength(self, wl):
        """
        Wavelength in nm.
        """
        time.sleep(self.delay)
        self.instr.write('WVL ' + str(float(wl)) + ' nm')

    def get_wavelength(self):
    	time.sleep(self.delay)
        return self.instr.query('WVL?')

    def open_shutter(self, tf):
        if tf == True:
            self.instr.write('D 0')
        else:
            self.instr.write('D 1')

        time.sleep(self.delay)

    def shutter_status(self):
        return self.instr.query('D?')

    def multimode_fiber(self, tf):
        if tf == True:
            self.instr.write('F 2')
        else:
            self.instr.write('F 1')

    def get_fiber_type(self):
        return self.instr.query('F?')

if __name__ == '__main__':
    x = EXFOFVA3100('GPIB0::10')
    x.setup_basic()
    x.set_wavelength(1550)
    x.set_attenuation(-1)
    print(x.get_wavelength())
