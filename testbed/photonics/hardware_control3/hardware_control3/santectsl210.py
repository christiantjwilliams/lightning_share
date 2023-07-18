import visa
import time

WL_MIN = 1500  # nm
WL_MAX = 1610  # nm
WL_STEP_MIN = 1e-4  # nm
WL_STEP_MAX = 160  # nm
SWEEP_SPEED_MIN = 0.5  # nm/s
SWEEP_SPEED_MAX = 5  # nm/s
SWEEP_SPEED_STEP = 0.1  # nm/s


class SantecTSL210(object):

    def __init__(self, address='GPIB0::1'):
        rm = visa.ResourceManager()
        self.instr = rm.open_resource(address)
        self.name = self.instr.query('*IDN?')

    def reset(self):
        self.instr.write('*RST')

    def setup_basic(self):
        # set basic settings
        self.set_output(output=True)
        self.set_wavelength(1550)

    def set_power(self, power, power_unit):
        """ Should be either dBm, mW, or uW """
        if power_unit == 'dBm':
            self.instr.write('OP %.2f' % power)
        elif power_unit == 'mW':
            self.instr.write('LP %.2f' % power)
        else:
            raise ValueError("Invalid power units")

    def set_wavelength(self, wav):
        # round to fourth decimal place
        self.instr.write('WA %.2f' % wav)

    def set_output(self, output=False):
        """
        Enable or disable laser diode.
        """

        if output is True:
            self.instr.write('LO')
        else:
            self.instr.write('LF')
        time.sleep(1)


class SantecLaser(SantecTSL210):
    pass

if __name__ == '__main__':
    obj = SantecTSL210()
    # obj.setup_basic()
    obj.set_wavelength(1550)
    obj.set_power(10,'dBm')
    # obj.set_output(True)

