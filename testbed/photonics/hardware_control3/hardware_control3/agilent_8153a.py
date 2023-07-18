import pyvisa as visa
import numpy as np
import time

MAXPOWER = 50e-3  # in Watts


class Agilent8153A(object):

    def __init__(self, address='GPIB0::21', dbm=False):
        rm = visa.ResourceManager()
        self.instr = rm.open_resource(address)
        self.name = self.instr.query('*IDN?')

        self.dbm = False

        self.setup_basic()

    def reset(self):
        self.instr.write('*RST')

    def setup_basic(self):
        self.instr.write('*RST')
        self.instr.write('INIT1:CONT 1')
        self.instr.write('INIT2:CONT 1')
        # Sets averaging time, 20ms < value < 3600s
        self.instr.write('SENS1:POW:ATIME .1')
        self.instr.write('SENS2:POW:ATIME .1')

        if self.dbm:
            self.instr.write('sens%d:pow:unit dbm' % (1))
            self.instr.write('sens%d:pow:unit dbm' % (2))
            time.sleep(1)

        self.set_wavelength(1560, 1)
        self.set_wavelength(1560, 2)

    def set_avg_time(self, t):
        self.instr.write('SENS{}:POW:ATIME {}'.format(1, t))
        self.instr.write('SENS{}:POW:ATIME {}'.format(2, t))

    def set_unit_dbm(self):
        self.instr.write('sens%d:pow:unit dbm' % (1))
        self.instr.write('sens%d:pow:unit dbm' % (2))

    def set_unit_watts(self):
        self.instr.write('sens%d:pow:unit watt' % (1))
        self.instr.write('sens%d:pow:unit watt' % (2))

    def set_autorange(self):
        self.instr.write('sens%d:pow:rang:auto 1' % (1))
        self.instr.write('sens%d:pow:rang:auto 1' % (2))

    def set_autorange_off(self):
        self.instr.write('sens%d:pow:rang:auto 0' % (1))
        self.instr.write('sens%d:pow:rang:auto 0' % (2))

    def set_upper_range(self, power_dBm, pd=1):
        self.instr.write('SENS%d:POW:RANG:UPPER %d' % (pd, power_dBm))

    def read_power(self, channel=1):
        time.sleep(.15)
        power_str = self.instr.query('READ{}:POW?'.format(channel))

        if self.dbm == True:
            maxPower = 10 * np.log10(MAXPOWER / 1e-3)
        else:
            maxPower = MAXPOWER

        # check for fake power reading
        while float(power_str) > maxPower:
            return -105.0
            print('Re-measuring')
            power_str = self.instr.query('READ{}:POW?'.format(channel))

        return float(power_str)  # Returns power in watts

    def set_continuous_trigger(self, b, chan=1):
        if b:
            self.instr.write('init{}:cont on'.format(chan))
        else:
            self.instr.write('init{}:cont off'.format(chan))

    def set_ratio_measurement(self, b):
        if b:
            self.instr.write('sens:pow:unit dbm')
            print(self.instr.query('sens:pow:unit?'))
            self.instr.write('sens2:pow:ref toref, 0dBm')
            print(self.instr.query('sens2:pow:ref? toref'))
            self.instr.write('sens1:pow:ref tob, 0dBm')
            print(self.instr.query('sens1:pow:ref? tob'))
            self.instr.write('sens1:pow:ref:stat:rati tob')
            print(self.instr.query('sens1:pow:ref:stat:rati?'))
            self.instr.write('sens1:pow:ref:disp')
            self.instr.write('sens1:pow:ref:stat on')
        else:
            self.instr.write('sens1:pow:ref toref, 0dBm')

    def set_wavelength(self, lambda_nm, channel=1):
        self.instr.write('SENS%d:POW:WAVE %0.6e' % (channel, lambda_nm * 1e-9))

    def get_wavelength(self, channel=1):
        return float(self.instr.query('sens%d:pow:wave?' % (channel))) * 1e9

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import time

    pm = Agilent8153A()
    print(pm.instr.query("*IDN?"))
    pm.setup_basic()
    pm.set_ratio_measurement(True)

    d = [(pm.read_power(), time.time()) for i in range(30)]
    y, x = list(zip(*d))
    y = np.array(y[1:])
    x = x[1:]

    pp = np.max(y) - np.min(y)
    print("Noise Level:", pp, "dB")
    plt.plot(x, y)
    plt.show()
