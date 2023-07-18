import visa

class Keithley2450(object):

    def __init__(self, address='USB0::0x05E6::0x2450::04028378::INSTR'):
        rm = visa.ResourceManager()
        self.instr = rm.open_resource(address)
        self.name = self.instr.query('*IDN?')
        self.delay = 0.001

    def reset(self):
        # reset the instrument
        self.instr.write('*RST')

    def setup_read_volt(self):
        # reset the instrument
        self.instr.write('*RST')

        # set to measure voltage with autorange enabled
        self.instr.write('SENS:VOLT:RANG:AUTO ON')

        # set to source current
        self.instr.write(':SOUR:FUNC CURR')

    def setup_read_curr(self):
        # reset the instrument
        self.instr.write('*RST')
        self.instr.write('*CLS')

        # self.instr.write(':TRAC:MAKE \"currBuffer\", 10')
        self.instr.write('SENS:FUNC \"CURR\"')

        # set to measure voltage with autorange enabled
        self.instr.write('SOUR:VOLT:RANG 0.02')
        self.instr.write('SOUR:VOLT 0')

        self.instr.write('SOUR:VOLT:ILIM 1')
        self.instr.write('SENS:CURR:RANG 1')
        self.instr.write('SENS:CURR:UNIT AMP')
        self.instr.write('SOUR:FUNC VOLT')


        self.instr.write('OUTP ON')
        self.instr.write('SENS:CURR:NPLC 0.2')
        self.instr.query('READ?')


    def enable_output(self, output=False):
        if output is True:
            self.instr.write(':OUTP ON')
        if output is False:
            self.instr.write(':OUTP OFF')

    def set_voltage(self, voltage=0e-6):
        # Set current level
        self.instr.write(':SOUR:VOLT:LEVEL %0.4e' % voltage)

    def read_voltage(self):
        return float(self.instr.query(':MEAS:VOLT?'))
        # print self.instr.query('SENS:CURR:READ?')

    def read_current(self):
        self.instr.query('MEAS:CURR?')
        return float(self.instr.query('READ?'))

    def source_v_read_i(self, vset, imax):
        self.instr.write('*RST')
        self.instr.write('SENS:FUNC "CURR"')
        self.instr.write('SENS:CURR:RANG:AUTO ON')
        self.instr.write('SENS:CURR:UNIT AMP')
        self.instr.write('SENS:CURR:OCOM ON')
        self.instr.write('SOUR:FUNC VOLT')

        vset_str = ('SOUR:VOLT %0.4f') % (vset)
        imax_str = ('SOUR:VOLT:ILIM %0.4f') % (imax)
        self.instr.write(vset_str)
        self.instr.write(imax_str)

        self.instr.write('COUNT 1')
        self.instr.write('OUTP ON')
        self.instr.write('TRAC:TRIG "defbuffer1"')

    # def read_curr(self):
    #     return self.instr.query('TRAC:DATA? 1, 1, "defbuffer1", SOUR, READ')


class KeithleySourceMeter(Keithley2450):
    pass

if __name__ == '__main__':
    # test code
    obj = Keithley2450()
    obj.setup_read_volt()
    obj.enable_output(True)
    while True:
        print(obj.read_voltage())
