import swabian
import swabian.swabian_client as client
import numpy as np
import itertools
import matplotlib.pyplot as plt
import time

class Swabian(object):
    def __init__(self):
        self.timeout = 100
        self.port = 36577

    def set_timeout(self, t):
        self.timeout = t

    def get_counts(self, chans=[1], binwidth=1e9, n=1000, debug=False):
        dbg = debug
        out = client.use_socket(function='count_trace',
                                arglist=[chans, binwidth, n],
                                keep_alive=False,
                                timeout=self.timeout,
                                debug=dbg)

        if len(chans)==1:
            return out['counts'][0]
        else:
            return out['counts']

    def get_coincidences(self, chans=[0,1,2,3], window=100, debug=False):
        dbg = debug

        coinGroups = []
        d = len(chans)
        for i in range(2, d+1):
            coinGroups += list(itertools.combinations(chans, i))

        out = client.use_socket(function='measure_coincidences',
                                arglist=[chans, coinGroups, window],
                                keep_alive=False,
                                timeout=self.timeout,
                                debug=debug)

        return out['counts']

if __name__=='__main__':
    swab = Swabian()
    print(swab.get_counts(chans=[1]))
