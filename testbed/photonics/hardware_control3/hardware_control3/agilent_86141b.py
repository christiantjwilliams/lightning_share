import visa
import numpy as np


class Agilent86141B(object):
    def __init__(self, address='GPIB0::12'):
        rm = visa.ResourceManager()
        self.instr = rm.open_resource(address)
        self.name = self.instr.query('*IDN?')

        self.setup_basic()

    def setup_basic(self):
        self.set_telecom_range()
        self.instr.write('SENS:BAND:RES 0.07 NM')

    def set_telecom_range(self):
        self.instr.query('SENS:WAV:CENT?')

        self.instr.write('SENS:WAV:CENT 1550 NM')
        self.instr.write('SENS:WAV:SPAN 60 NM')

    def set_res_bw(self, bw): # in nm
        self.instr.write('SENS:BAND:RES %s NM' % bw)

    def set_span(self, wl_start, wl_stop): # in nm
        center = np.round(0.5*(wl_start+wl_stop), 3)
        span = np.round(wl_stop - wl_start, 3)
        self.instr.write('SENS:WAV:CENT %s NM' % center)
        self.instr.write('SENS:WAV:SPAN %s NM' % span)

    def take_sweep(self):
        self.instr.write('INIT:IMM')

    def get_trace(self, trace='A'):
        start_wl = float(self.instr.query('TRACE:X:START? TR%s' % trace.upper())) * 1e9
        stop_wl = float(self.instr.query('TRACE:X:STOP? TR%s' % trace.upper())) * 1e9
        num_pts = float(self.instr.query('TRACE:POINTS? TR%s' % trace.upper()))

        pwr = self.instr.query('TRACE:DATA:Y? TR%s' % trace.upper())
        pwr = pwr.split(',')
        pwr = [float(y) for y in pwr]

        return np.linspace(start_wl, stop_wl, num_pts), np.array(pwr)

    def set_peak_to_center(self):
        self.instr.write('CALC:MARK1:MAX')
        self.instr.write('CALC:MARK1:SCEN')

    def get_center_wl(self):
        return float(self.instr.query('SENS:WAV:CENT?')) * 1e6

    def start_filter_mode(self):
        self.instr.write('INST:SEL "FILTER"')

    def set_filter_center_wavelength(self, wav_nm):
        set_wl = int(wav_nm*1000)
        self.instr.write('INP:FILT:X %sPM' % set_wl)

    def measure_peak_wl(self):
        self.set_telecom_range()
        self.get_sweep()
        self.set_peak_to_center()
        return self.get_center_wl()

if __name__ == '__main__':
    osa = Agilent86141B()
