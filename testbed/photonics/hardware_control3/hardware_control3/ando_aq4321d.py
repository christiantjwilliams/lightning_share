import visa


class AndoAQ4321D(object):
    def __init__(self, address='GPIB0::24'):
        rm = visa.ResourceManager()
        self.instr = rm.open_resource(address)
        self.name = self.instr.write('*IDN?')
        self.delay = 0.1
        self.lambda_offset = -0.045 #this is the offset in nm of the laser at 1550nm

    def reset(self):
        self.instr.write('*RST')

    def setup_basic(self):
        self.instr.write('TPOU0')  # Sets power unit to dBm
        self.instr.write('TWLFRU0')  # Sets wl unit to nm
        self.set_wavelength(1550.000)

    def set_power_unit(self, power_unit):
        """ Should be either dBm (0), mW (1) """
        self.instr.write('TPOU %s' % power_unit)

    def optical_off(self):
        """ turns optical output off"""
        self.instr.write('L0')

    def optical_on(self):
        """ turns optical output off"""
        self.instr.write('L1')

    def set_power(self, power):
        self.instr.write('TPDB{}'.format(power))

    def get_power(self):
        power_str = self.instr.query('TPDB?')
        return str(power_str)

    def set_wavelength(self, lambda_nm):
        self.instr.write('TWL %0.3f' % (lambda_nm+self.lambda_offset))

    def get_wavelength(self):
        wavelength_str = self.instr.query('TWL?')
        return float(wavelength_str)


class AndoLaser(AndoAQ4321D):
    pass

if __name__ == '__main__':
    obj = AndoAQ4321D('GPIB0::24')
    obj.optical_on()
    obj.set_wavelength(1550)
    obj.set_power(0.)
    print(obj.get_wavelength())
    print(obj.get_power())
