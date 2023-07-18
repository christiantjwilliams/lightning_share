import visa
import time


class AgilentE3631(object):

    def __init__(self, address='GPIB0::06'):
        rm = visa.ResourceManager()
        self.instr = rm.open_resource(address)
        self.name = self.instr.query('*IDN?')
        self.delay = 0.1

    def reset(self):
        self.instr.write('*CLS')
        self.instr.write('*RST')

    def setup_basic(self):
        pass

    def get_sourced_current(self):
        reps = 1
        for i in range(1):
            # measure current output currents
            plus_term = self.instr.query('MEAS:CURR:DC? P25V')
            minus_term = self.instr.query('MEAS:CURR:DC? N25V')

        return float(plus_term), float(minus_term)

    def set_output_levels(self, supply, v_lim, i_lim):
        """
        Set desired output level.
        """

        cmd = 'APPL %s, %s, %s' % (supply, v_lim, i_lim)
        self.instr.write(cmd)

    def enable(self, b):
        if b:
            self.instr.write('OUTP:STATE ON')
        else:
            self.instr.write('OUTP:STATE OFF')

    def is_enabled(self):
        self.instr.write('OUTP:STATE?')
        if int(self.instr.read()[:-1]) == 1:
            return True
        else:
            return False


class AgilentPowerSupply(AgilentE3631):
    pass

if __name__ == '__main__':
    obj = AgilentE3631()
    obj.reset()
    obj.enable(True)
    # import time
    # start = time.time()
    print(obj.get_sourced_current())
    # print time.time() - start
    # obj.set_output_levels('P25V', 15., .01)
    # obj.set_output_levels('N25V', -8., .01)
