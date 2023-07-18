import visa
import time

class Keithley2400(object):

    """
    Python wrapper for Keithley2400 DC Power Supply
    """

    def __init__(self, address, timeout=10000):
        rm = visa.ResourceManager()
        self.instr = rm.open_resource(address)
        self.name = self.instr.query('*IDN?')
        self.__timeout = timeout
        self.reset()
        self.clearall()
        time.sleep(.1)

    def reset(self):
        self.instr.write('*RST')
        self.instr.timeout = self.__timeout

    def clearall(self):
        self.instr.write('*CLS')
        self.instr.timeout = self.__timeout

    def set_output(self,output=False):
        if output:
            self.instr.write('output on')
        else:
            self.instr.write('output off')

    def __read(self):
        read_str = self.instr.query('read?')
        # See page 18-51 of manual, returns: voltage, current, resistance, timestamp, status info
        # Returns something like
        # '5.275894E-05,-1.508318E-06,+9.910000E+37,+2.562604E+03,+3.994000E+04'
        data = read_str.split(',')
        voltage = float(data[0])
        current = float(data[1])
        resistance = float(data[2])
        return voltage, current, resistance

    def enable_live_readings(self):
        self.instr.write('ARM:COUN INF')
        self.instr.write('TRIG:DEL 0')
        self.instr.write('INIT')

    def setup_measure_voltage(self):
        self.instr.write("SOUR:FUNC CURR")
        self.instr.write("SOUR:CURR 0")
        self.instr.write("CONF:VOLT")
        self.instr.write("FORM:ELEM VOLT")
        time.sleep(1)
        
    def setup_v_meas(self,term='FRON'):
        # choose ports
        self.instr.write(':ROUT:TERM '+term)
        # set high sensitivity
        self.instr.write(':SENS:VOLT:NPLC 1')

    def read_voltage(self):
        data = self.instr.query(':MEAS:VOLT?')
        data = data.split(',')
        data = list(map(float, data))
        return data[0]

    def __setup_source_measure(self,compliance,\
        source,measure,fourWire=False,sourceRange=None,measureRange=None):
        self.reset()
        self.instr.write('source:function '+source)
        self.instr.write('source:'+source+':mode fixed')

        self.instr.write('source:'+source+':range:auto on')
        if(sourceRange is None):
            self.instr.write('source:'+source+':range default')
        else:
            self.instr.write('source:'+source+':range '+str(sourceRange))

        self.instr.write('source:'+source+':level 0')

        if fourWire:
            self.instr.write('system:rsense on') # turn on 4-wire remote sensing
        else:
            self.instr.write('system:rsense off')

        self.instr.write('sense:function \"'+measure+'\"')
        self.instr.write('sense:'+measure+':protection '+str(compliance))

        self.instr.write('sense:'+measure+':range:auto on')
        if(measureRange is None):
            self.instr.write('sense:'+measure+':range '+str(compliance))
        else:
            self.instr.write('sense:'+measure+':range '+str(measureRange))

        self.instr.write('output on')
        pass

    def setup_sourceVoltage_measureCurrent(self,currentCompliance,fourWire=False,\
        voltageRange=None,currentRange=None):
        self.__setup_source_measure(currentCompliance,'voltage','current',\
            fourWire,voltageRange,currentRange)
        pass

    def setup_sourceCurrent_measureVoltage(self,voltageCompliance,fourWire=False,\
        currentRange=None,voltageRange=None):
        self.__setup_source_measure(voltageCompliance,'current','voltage',\
            fourWire,currentRange,voltageRange)
        pass

    def set_current(self,current=0):
        self.instr.write('source:current:level '+str(current))

    def set_voltage(self,voltage=0):
        self.instr.write('source:voltage:level '+str(voltage))

    def get_current(self):
        voltage, current, resistance = self.__read()
        return current

    def get_voltage(self):
        voltage, current, resistance = self.__read()
        return voltage

    def get_resistance(self):
        voltage, current, resistance = self.__read()
        return resistance
