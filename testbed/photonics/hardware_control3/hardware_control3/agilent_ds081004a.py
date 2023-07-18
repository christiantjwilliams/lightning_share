import visa
import numpy as np
import struct


class AgilentDS081004A(object):

    """
    Python wrapper for AgilentDS081004A oscilloscope
    """

    def __init__(self, address, timeout=10000):
        rm = visa.ResourceManager()
        self.instr = rm.open_resource(address)
        self.name = self.instr.query('*IDN?')
        self.instr.write('SYST:HEAD OFF')
        self.__timeout = timeout
        self.instr.timeout = self.__timeout

    def __value_limits(self, val, minVal, maxVal):
        if(val < minVal or val > maxVal):
            raise ValueError("improper input value")
        else:
            return val

    def __check_options(self, option, acceptedOptions):
        if option.lower() not in acceptedOptions:
            raise ValueError("no such option")
        else:
            return option

    def _build_ieee_block(self, data):
        "Build IEEE block"
        # IEEE block binary data is prefixed with #lnnnnnnnn
        # where l is length of n and n is the
        # length of the data
        # ex: #800002000 prefixes 2000 data bytes
        return str('#8%08d' % len(data)).encode('utf-8') + data

    def _decode_ieee_block(self, data):
        "Decode IEEE block"
        # IEEE block binary data is prefixed with #lnnnnnnnn
        # where l is length of n and n is the
        # length of the data
        # ex: #800002000 prefixes 2000 data bytes
        if len(data) == 0:
            return b''

        ind = 0
        c = '#'.encode('utf-8')
        while data[ind:ind + 1] != c:
            ind += 1

        ind += 1
        l = int(data[ind:ind + 1])
        ind += 1

        if (l > 0):
            num = int(data[ind:ind + l].decode('utf-8'))
            ind += l

            return data[ind:ind + num]
        else:
            return data[ind:]

    def reset(self):
        self.instr.write('*RST')
        self.instr.timeout = self.__timeout

    def clearall(self):
        self.instr.write('*CLS')
        self.instr.timeout = self.__timeout

    def autoscale(self):
        self.instr.write('autoscale')

    def setup_basic(self):  # setup basic settings
        self.reset()
        self.clearall()
        self.instr.write('acquire:complete 100')
        for i in range(4):
            self.instr.write('CHAN' + str(i + 1) + ':INPUT DC50')
            self.set_chan_units(i + 1, 'volt')

    def setup_averaging_function(self, fn, chan, avgNum):
        self.instr.write('FUNC' + str(fn) + ':AVERAGE CHAN' +
                         str(chan) + ',' + str(avgNum))
        self.instr.write('FUNC' + str(fn) + ':DISPLAY ON')

    def check_function(self, fn):
        return self.instr.query('FUNC' + str(fn) + '?')

    def setup_differentiate_function(self, fn, chan):
        self.instr.write('FUNC' + str(fn) + ':DIFF CHAN' + str(chan))
        self.instr.write('FUNC' + str(fn) + ':DISPLAY ON')

    def setup_integrate_function(self, fn, chan):
        self.instr.write('FUNC' + str(fn) + ':INT CHAN' + str(chan))
        self.instr.write('FUNC' + str(fn) + ':DISPLAY ON')

    """
    CHANNEL FUNCTIONS (FOR Y-AXIS RELATED MANIPULATION)
    """

    def set_chan_display(self, chan, disp='on'):
        self.instr.write('channel' + str(chan) + ':display ' + disp)

    def get_chan_display(self, chan):
        val = self.instr.query('channel' + str(chan) + ':display?')
        if(val is 1):
            return 'on'
        elif(val is 0):
            return 'off'

    def set_chan_units(self, chan, unit='VOLT'):
        if(unit.lower() == 'volt' or unit.lower() == 'amp' or unit.lower() == 'watt'):
            self.instr.write('CHAN' + str(chan) + ':UNITS ' + unit.lower())
        else:
            print('Unknown units')

    def get_chan_units(self, chan):
        unit = self.instr.query('CHAN' + str(chan) + ':UNITS?')
        return unit

    def set_chan_offset(self, chan, offset):
        self.instr.write('CHAN' + str(chan) + ':OFFSET ' + str(offset))

    def get_chan_offset(self, chan):
        offset = self.instr.query('CHAN' + str(chan) + ':OFFSET?')
        return float(offset)

    def set_chan_scale(self, chan, scale):  # scale is units per division
        self.instr.write('CHAN' + str(chan) + ':SCALE ' + str(scale))

    def get_chan_scale(self, chan):
        scale = self.instr.query('CHAN' + str(chan) + ':SCALE?')
        return float(scale)

    def set_chan_range(self, chan, chanRange):  # range is 10 times the scale (above)
        self.instr.write('CHAN' + str(chan) + ':RANGE ' + str(chanRange))

    def get_chan_range(self, chan):
        chanRange = self.instr.query('CHAN' + str(chan) + ':RANGE?')
        return float(chanRange)
    """
    TIMEBASE FUNCTIONS (FOR X-AXIS RELATED MANIPULATION)
    """

    def set_time_scale(self, scale):  # scale is in seconds per division
        self.instr.write('timebase:scale ' + str(scale))

    def get_time_scale(self):
        scale = self.instr.query('timebase:scale?')
        return float(scale)

    def set_time_range(self, timeRange):  # range is 10 times the scale
        self.instr.write('timebase:range ' + str(timeRange))

    def get_time_range(self):
        timeRange = self.instr.query('timebase:range?')
        return float(timeRange)

    def get_waveform(self, source, sourceNum):
        self.instr.write('waveform:byteorder msbfirst')
        self.instr.write('waveform:format word')
        self.instr.write('waveform:source ' + source + str(sourceNum))

        # Read preamble
        pre = self.instr.query('waveform:preamble?')
        pre = pre.split(',')

        format = int(pre[0])
        # acquisitionType = int(pre[1])
        points = int(pre[2])
        # count = int(pre[3])
        xInc = float(pre[4])
        xOrg = float(pre[5])
        xRef = int(float(pre[6]))
        yInc = float(pre[7])
        yOrg = float(pre[8])
        yRef = int(float(pre[9]))

        if format != 2:
            print('unexpected response')
            raise

        self.instr.write('waveform:data?')
        # read binary data from the instrument
        raw_data = self._decode_ieee_block(self.instr.read_raw())
        # sort the data out
        xData = np.empty([points])
        yData = np.empty([points])
        for i in range(points):
            xData[i] = ((i - xRef) * xInc) + xOrg
            yval = struct.unpack(">h", raw_data[i * 2:i * 2 + 2])[0]
            if yval == 31232:
                # hole value
                yData[i] = float('nan')
            else:
                yData[i] = ((yval - yRef) * yInc) + yOrg

        return xData, yData

    def measure_period(self, source, sourceNum):
        val = self.instr.query('measure:period? ' + source + str(sourceNum))
        return float(val)

    def measure_frequency(self, source, sourceNum):
        val = self.instr.query('measure:frequency? ' + source + str(sourceNum))
        return float(val)

    def measure_vamplitude(self, source, sourceNum):
        val = self.instr.query('measure:vamplitude? ' + source + str(sourceNum))
        return float(val)

    def measure_vpp(self, source, sourceNum):
        val = self.instr.query('measure:vpp? ' + source + str(sourceNum))
        return float(val)

    def measure_vmax(self, source, sourceNum):
        val = self.instr.query('measure:vmax? ' + source + str(sourceNum))
        return float(val)

    def measure_vmin(self, source, sourceNum):
        val = self.instr.query('measure:vmin? ' + source + str(sourceNum))
        return float(val)

    def measure_vaverage(self, source, sourceNum):
        val = self.instr.query(
            'measure:vaverage? display,' + source + str(sourceNum))
        return float(val)
    """
    FUNCTIONS RELATED TO TRIGGER
    """

    def setup_trigger_basic(self, channel, level):
        self.set_trigger_mode('edge')
        self.set_trigger_source(channel)
        self.set_trigger_edge_slope('positive')
        self.set_trigger_level(channel, level)

    def set_trigger_mode(self, mode):
        acceptedModes = ['edge', 'glitch', 'glit']
        try:
            self.instr.write('trigger:mode ' +
                             self.__check_options(mode, acceptedModes))
        except ValueError:
            print('advanced mode is not yet supported, only edge or glitch')

    def get_trigger_mode(self):
        return self.instr.query('trigger:mode?')

    def set_trigger_time_holdoff(self, holdoffTime):
        try:
            self.instr.write('trigger:holdoff ' +
                             str(self.__value_limits(holdoffTime, 80e-9, 320e-3)))
        except ValueError:
            print('Check that hold off time can only be between 80ns and 320ms')

    def get_trigger_time_holdoff(self):
        return float(self.instr.query('trigger:holdoff?'))

    def set_trigger_high_threshold(self, channel, highThreshold):
        self.instr.write('trigger:hthreshold channel' + str(channel) +
                         ',' + str(highThreshold))

    def get_trigger_high_threshold(self, channel):
        return float(self.instr.query('trigger:hthreshold? channel' + str(channel)))

    def set_trigger_low_threshold(self, channel, lowThreshold):
        self.instr.write('trigger:lthreshold channel' + str(channel) +
                         ',' + str(lowThreshold))

    def get_trigger_low_threshold(self, channel):
        return float(self.instr.query('trigger:lthreshold? channel' + str(channel)))

    def set_trigger_hysteresis(self, hyst):
        acceptedOptions = ['normal', 'hsensitivity', 'norm', 'hsen']
        try:
            self.instr.write('trigger:hysteresis '
                             + self.__check_options(hyst, acceptedOptions))
        except ValueError:
            print(('choices are:', acceptedOptions))

    def get_trigger_hysteresis(self):
        return self.instr.query('trigger:hysteresis?')

    def set_trigger_level(self, channel, level):
        self.instr.write('trigger:level channel' + str(channel) +
                         ',' + str(level))

    def get_trigger_level(self, channel):
        return float(self.instr.query('trigger:level? channel' + str(channel)))

    def set_trigger_sweep(self, sweep):
        acceptedOptions = ['auto', 'triggered', 'trig', 'single', 'sing']
        try:
            self.instr.write('trigger:sweep '
                             + self.__check_options(sweep, acceptedOptions))
        except ValueError:
            print(('choices are:', acceptedOptions))

    def get_trigger_sweep(self):
        return self.instr.query('trigger:sweep?')

    def set_trigger_edge_slope(self, slopeSign):
        acceptedOptions = ['positive', 'negative', 'pos', 'neg']
        if(self.get_trigger_mode().rstrip('\n') is 'edge'):
            try:
                self.instr.write('trigger:edge:slope ' +
                                 self.__check_options(slopeSign, acceptedOptions))
            except ValueError:
                print(('choices are:', acceptedOptions))
        else:
            pass

    def get_trigger_edge_slope(self):
        if(self.get_trigger_mode().rstrip('\n') is 'edge'):
            self.instr.query('trigger:edge:slope?')
        else:
            pass

    def set_trigger_source(self, channel):
        mode = self.get_trigger_mode().rstrip('\n')
        self.instr.write('trigger:' + mode + ':source channel' + str(channel))

    def get_trigger_source(self):
        mode = self.get_trigger_mode().rstrip('\n')
        return self.instr.query('trigger:' + mode + ':source?')
