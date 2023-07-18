import pyvisa as visa

class JDSUHA9(object):

    def __init__(self, address='GPIB0::5'):
        rm = visa.ResourceManager()
        self.instr = rm.open_resource(address)
        #self.name = self.instr.query('*IDN?')
        self.delay = 0.1

    def reset(self):
        self.instr.write('RESET')
        # nominally goes to 1310nm, so set to 1550nm
        self.instr.write('WVL 1550 nm')

    def setup_basic(self):
        self.reset()
        self.instr.write('DISP0')  # attenuation mode

    def set_wavelength(self, wl):
        """
        wl in [nm]
        """
        self.instr.write('WVL %f NM' % (wl))

    def set_attenuation(self, a):
        """
        'a' in [dB]
        """
        self.instr.write('CLR')
        self.instr.write('SRE 4')
        ret = self.instr.query('ATT %f dB;CNB?' % (abs(a)))
        if ret != 4:
            self.wait_for_movement()
        #self.instr.wait_for_srq()

    def block_beam(self, tf):
        """
        Block the beam.
        """
        if tf == True:
            self.instr.write('D1')
        else:
            self.instr.write('D0')

    def driver_toggle(self, tf):
        """
        Disable or enable the driver.
        """

        if tf == True:
            self.instr.write('XDR0')
        else:
            self.instr.write('XDR1')

    def get_attenuation(self):
        attenuation_str = self.instr.query('ATT?')
        return float(attenuation_str)  # Returns attenuation in [dB]

    def get_wavelength(self):
        wvl = self.instr.query('WVL?')
        return float(wvl)

    def wait_for_movement(self):
        moving = True
        while moving:
            ret = self.instr.query('CNB?')
            if int(ret) == 4:
                moving = False


if __name__ == '__main__':
    obj = JDSUHA9('GPIB0::7')
    #obj.set_wavelength(1560)
    #print obj.get_wavelength()
    obj.set_attenuation(10.)
    print(obj.get_attenuation())
    # obj.block_beam(False)
