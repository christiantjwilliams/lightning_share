import visa
import numpy as np
from scipy.interpolate import interp1d

class AnritsuMG9638A(object):
    def __init__(self, address='GPIB0::20'):
        rm = visa.ResourceManager()
        self.instr = rm.open_resource(address)
        self.name = self.instr.query('*IDN?')
        self.delay = 0.1

    def reset(self):
        self.instr.write('*RST')

    def setup_basic(self):
        self.instr.write('MCW')  # Set laser to CW mode
        self.instr.write('COH ON')  # Set laser to coherent mode
        self.instr.write('POWU dBm')  # Sets power unit to dBm
        self.set_wavelength(1550)

    def set_power_unit(self, power_unit):
        """ Should be either dBm, mW, or uW """
        self.instr.write('POWU %s' % power_unit)

    def get_power_unit(self, power_unit):
        return self.instr.query('POWU?')

    def get_wavelength(self):
        wavelength_str = self.instr.query('OUTW?')
        return float(wavelength_str)*1e9

    def set_output(self, output=False):
        self.instr.write('OUTP %d' % output)

    def get_output(self):
        return bool(self.instr.query('OUTP?'))

    def get_power(self):
        power_str = self.instr.query('POW?')
        return float(power_str)

    def set_wavelength(self, lambda_nm):
        self.instr.write('WCNT %0.3fNM' % (lambda_nm))

    def set_power(self, power, power_unit='dBm'):
        self.instr.write('POW {}{}'.format(power, power_unit))

    def coh_ctrl_on(self):
        self.instr.write('COH ON')

    def coh_ctrl_off(self):
        self.instr.write('COH OFF')

class AnritsuLaser(AnritsuMG9638A):
    pass

if __name__ == '__main__':
    import time

    obj = AnritsuMG9638A('GPIB0::20')
    obj.set_wavelength(1550.)
    print(obj.get_wavelength())
    obj.set_wavelength(1550.+1.0)
    time.sleep(1)
    print(obj.get_wavelength())
    obj.set_power(-5.)
    obj.coh_ctrl_off()
