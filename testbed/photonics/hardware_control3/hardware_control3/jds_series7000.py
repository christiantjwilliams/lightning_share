import visa
import time

class JDSSeries7000(object):
    
    def __init__(self, address='GPIB::07'):
        rm = visa.ResourceManager()
        self.instr = rm.open_resource(address)
        self.name = self.instr.query('*IDN?')
        self.delay = 0.1
    
    def reset(self):
        self.instr.write('RESET')
        # nominally goes to 1310nm, so set to 1550nm
        #self.instr.write('WVL 1550 nm') 
        

if __name__ == '__main__':
    obj = JDSSeries7000()
    obj.reset()
