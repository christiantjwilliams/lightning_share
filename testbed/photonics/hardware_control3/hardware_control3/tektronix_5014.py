import visa
import numpy as np
import ctypes
from struct import pack
import matplotlib.pyplot as plt

from threading import Thread
from multiprocessing import Pool, Process
from qcodes.instrument_drivers.tektronix.AWGFileParser import parse_awg_file
from qcodes.instrument_drivers.tektronix.AWG5014 import Tektronix_AWG5014
from qcodes.logger.logger import start_all_logging
from qcodes.dataset.plotting import plot_dataset
from qcodes import (
    Measurement,
    experiments,
    initialise_database,
    initialise_or_create_database_at,
    load_by_guid,
    load_by_run_spec,
    load_experiment,
    load_last_experiment,
    load_or_create_experiment,
    new_experiment,
)
import qcodes as qc
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import time
import os
import sys
import nidaqmx as daq
from nidaqmx.constants import AcquisitionType
from nidaqmx.constants import Edge


class TektronixAWG5014(object):
    def __init__(self):
        station = qc.Station()
        # self.awg = Tektronix_AWG5014("AWG5014", "GPIB0::5::INSTR") #If you are using GPIB to connect to the AWG
        self.awg = Tektronix_AWG5014(
            "AWG5014", "TCPIP0::169.254.221.85::inst0::INSTR") #If you connect to the AWG over ethernet. Much faster writing times than GPIB!
        
        station.add_component(self.awg)

    '''
    This AWG has a DC voltage port on the bottom right of the box. Here is a quick method for using it!
    '''
    def set_dc_voltage(self,voltage,channel=1):
        if channel == 1:
            self.awg.ch1_DC_out.set(voltage)
        elif channel == 2:
            self.awg.ch2_DC_out.set(voltage)
        elif channel == 3:
            self.awg.ch3_DC_out.set(voltage)
        elif channel == 4:
            self.awg.ch4_DC_out.set(voltage)
        else:
            print("Set DC Channel not recognized")

    def reset_dc_voltages(self):
        self.set_dc_voltage(0,1)
        self.set_dc_voltage(0,2)
        self.set_dc_voltage(0,3)
        self.set_dc_voltage(0,4)

    '''
    Method for taking four waveforms and writing them out all at once.
    A single trigger is output on the marker chanenls when the waveforms are finished.
    Make sure to call run_awg to actually run this!
    '''
    def fast_waveform_write(self,waveform1,waveform2,waveform3,waveform4):
        waveforms = [[], [], [], []]
        m1s = [[], [], [], []]
        m2s = [[], [], [], []]

        self.num_triggers = 1

        waveforms[0].append(waveform1)
        waveforms[1].append(waveform2)
        waveforms[2].append(waveform3)
        waveforms[3].append(waveform4)

        waveform_length = len(waveform1)

        for i in range(4):
            m1 = np.zeros(waveform_length)
            offset = waveform_length - 160 #Change this to change the size of the pulse. I was running at 10MHz so 160 -> 16us TTL pulse.
            for j in range(self.num_triggers):
                m1[offset:offset + 160] = 1
            m1s[i].append(m1)
            m2 = np.zeros(waveform_length)
            m2s[i].append(m2)
        
        self.write_waveforms(waveforms,m1s,m2s)
        

    '''
    Takes in an array of the sequences to put on the awg channels. A sequence is a number of waveforms which we put out one after the other.
    Also takes markers (triggers) which are output at the same time as the waveforms.
    '''
    def write_waveforms(self,waveforms,m1s,m2s):
        # Sequencing options
        num_sequences = len(waveforms[0])
        # number of repetitions
        nreps = [1 for i in range(num_sequences)]
        # Wait trigger (0 or 1)
        trig_waits = [0 for i in range(num_sequences)]
        # Goto state
        goto_states = [((ii+1) % num_sequences)+1 for ii in range(num_sequences)]
        # goto_states = [0]
        # Event jump
        jump_tos = [2 for i in range(num_sequences)]
        
        #Then set the sequence length to the correct number lest not all sequence elements get uploaded
        self.awg.sequence_length.set(num_sequences)

        #Then transfer the waveforms to the list...
        for elnum in range(num_sequences):
            for chnum in range(4):
                wfmname = 'wfm{:03d}ch{}'.format(elnum, chnum+1)
                self.awg.send_waveform_to_list(
                    waveforms[chnum][elnum], m1s[chnum][elnum], m2s[chnum][elnum], wfmname)
        #...upload them to the sequencer...
        for elnum in range(num_sequences):
            for chnum in range(4):
                wfmname = 'wfm{:03d}ch{}'.format(elnum, chnum+1)
                self.awg.set_sqel_waveform(wfmname, chnum+1, elnum+1)
        #...and set the sequence elements setting
        for elnum in range(num_sequences):
            self.awg.set_sqel_goto_target_index(elnum+1, goto_states[elnum])
            self.awg.set_sqel_loopcnt(nreps[elnum], elnum+1)
            self.awg.set_sqel_trigger_wait(elnum+1, trig_waits[elnum])
            self.awg.set_sqel_event_target_index(elnum+1, jump_tos[elnum])

    def run_awg(self):

        self.awg.all_channels_on()
        time.sleep(0.2)
        self.awg.run()

    def reset_waveforms(self):
        # First clear up the waveform list and empty the sequencer
        self.awg.delete_all_waveforms_from_list()

    def __del__(self):
        self.awg.close()


if __name__ == "__main__":
    awg = TektronixAWG5014()
    t = np.linspace(0,1,num=1000)
    sinewave = np.sin(2*np.pi*10*t) #10 sinewave periods
    awg.fast_waveform_write(sinewave,sinewave,sinewave,sinewave) #Put a sinewave on all channels at once!

