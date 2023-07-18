import numpy as np
import time
import sys
sys.path.append(r'C:\Users\QPGroup\Documents\git\drvr')
from drvr.lightgrid2_wrapper import lightGridWrapper

class LightGrid(object):

    def __init__(self, avgs=100, convert=True):
        self.pds = lightGridWrapper()
        self.avgs = avgs
        self.convert = convert

    def set_avgs(self, avgs):
        self.avgs = avgs

    def read_list(self,chan_list=list(range(18))):
        pows = np.array(self.pds.read(N=self.avgs, convert=self.convert))

        pow_list = list(pows[chan_list])
        return pow_list

    def read_power(self, channel=1):
        pows = np.array(self.pds.read(N=self.avgs, convert=self.convert))
        chan_pow = pows[channel]

        if chan_pow <0.0:
            return 0.0
        else:
            return chan_pow

    def live_plot(self, chan_list=None):
        self.pds.live_plot(chan_list)



if __name__ == '__main__':

    x = LightGrid()
    print(x.read_power())
