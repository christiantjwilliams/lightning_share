import visa
import numpy as np

class Hp8270b(object):
    """
    Python wrapper for HP8270B Vector Network Analyzer
    """

    def __init__(self, address,timeout=10000):
        rm = visa.ResourceManager()
        self.instr = rm.open_resource(address)
        self.instr.write('CHAN2;') # make channel 2 active because chan. 1 is broken
        self.name = self.instr.query('*IDN?')
        self.__timeout = timeout
        self.instr.timeout = self.__timeout

    def reset(self):
        self.instr.write('*RST;')
        self.instr.write('CHAN2;') # make channel 2 active because chan. 1 is broken
        self.instr.timeout = self.__timeout
        self.instr.delay = .1

    def clearall(self):
        self.instr.write('*CLS;')
        self.instr.timeout = self.__timeout
        self.instr.delay = .1

    def fixed_freq(self, f=10e6):
        self.instr.write('CWFREQ%0.6eHZ;' % f)

    def set_freq_range(self,f_start=0.1e9, f_stop=1.0e9, num_pts=400):
        self.instr.write('POIN;%0.0f;' % num_pts) # doesn't work well, seems to only accept specific values
        self.instr.write('STAR;%0.6e;' % f_start)
        self.instr.write('STOP;%0.6e;' % f_stop)

    def power(self, power=-10):
        # if (power < -60) or (power > 0):
        #     print 'Out of range value'
        #     return
        # if power >= -5:
        #     power_range = '01'  # Only available below 26 GHz
        # elif power >= -20:
        #     power_range = '02'
        # elif power >= -35:
        #     power_range = '05'
        # elif power >= -50:
        #     power_range = '08'
        # elif power >= -60:
        #     power_range = '10'

        # self.instr.write('PRAN%s' % power_range)
        self.instr.write('POWE;%0.0d' % power)  # Sets power (in dBm)

    def s_mode(self, s_mode='S11'):
        self.instr.write('%s;' % s_mode)

    def format_polar(self):
        self.instr.write('POLA;')  # Set to polar coordinates

    def format_logarithmic(self):
        self.instr.write('LOGM;')  # Set to polar coordinates

    def run_sweep_ri(self):
        """ Runs a sweep using whatever settings are currently on the NA and returns the real
        and imaginary components of each data point """

        self.format_polar()  # Set to polar coordinates
        f_start = float(self.instr.query('STAR?;'))
        f_stop = float(self.instr.query('STOP?;'))
        num_pts = int(float(self.instr.query('POIN?;')))

        print('Sweeping from %0.0d GHz to %0.0d GHz, with %d points' % (f_start / 1e9, f_stop / 1e9, num_pts))
        self.instr.write('SING;')
        # # while (self.instr.no bits are waiting):
        # #     sleep(1)
        # #     print 'Operation not yet complete, waiting 1s'

        self.instr.write('FORM4;')  # Make the data output in ASCII
        raw_data = self.instr.query('OUTPFORM;')
        
        # post-process the data here
        raw_data = raw_data.split("\n")
        raw_data = np.array(raw_data)
        R = []
        I = []
        # now each line of raw data contains the real number and imag number
        # separated by a comma. The for loop for raw_data must be specified
        # because there are blanks (which cannot be converted into a float)
        # at the last line.
        for line in raw_data[0:num_pts:1]:
            split_line = line.split(",")
            R.append(float(split_line[0]))
            I.append(float(split_line[1]))

        f = np.linspace(f_start, f_stop, num_pts)
        return f, R, I
   
    

