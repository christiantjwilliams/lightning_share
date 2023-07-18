#This is a wrapper around the qcodes driver for the keysight chasis

import numpy as np

try:
    import keysightSD1
except:
    # add the path where the keysight library probably resides and try again
    import sys
    sys.path.append('C:\Program Files (x86)\Keysight\SD1\Libraries\Python')
    import keysightSD1

import qcodes
from qcodes_contrib_drivers.drivers.Keysight.M3202A import M3202A

# try to close station from previous run.
try:
    station.close_all_registered_instruments()
except:
    pass

class keysight():
    def __init__(self,cyclic=False):
        self.cyclic = cyclic
        self.setup(cyclic)

    def setup(self,cyclic=False):
        self.awg1 = M3202A("AWG1", chassis = 1, slot = 2)
        self.awg2 = M3202A("AWG2", chassis = 1, slot = 3)
        self.awg3 = M3202A("AWG3", chassis = 1, slot = 4)
        self.awg4 = M3202A("AWG4", chassis = 1, slot = 5)

        self.collection_of_waveform_references = []

        station = qcodes.Station()

        station.add_component(self.awg1)
        station.add_component(self.awg2)
        station.add_component(self.awg3)
        station.add_component(self.awg4)

        # setup output channels
        pxi1 = keysightSD1.SD_TriggerExternalSources.TRIGGER_PXI1
        trigger_mode = keysightSD1.SD_TriggerBehaviors.TRIGGER_FALL

        for awg in [self.awg1, self.awg2, self.awg3, self.awg4]:
            for ch in range(1,5):
                awg.set_channel_offset(0.0, ch)
                awg.set_channel_amplitude(1.5, ch)

                awg.set_channel_wave_shape(keysightSD1.SD_Waveshapes.AOU_AWG, ch)
                if cyclic == True:
                    awg.awg_queue_config(ch, keysightSD1.SD_QueueMode.CYCLIC)
                elif cyclic == False:
                    awg.awg_queue_config(ch, keysightSD1.SD_QueueMode.ONE_SHOT)
                awg.awg_config_external_trigger(ch, pxi1, trigger_mode)

    def set_dc_output(self,ch1,input_mzi):
        self.awg1.set_channel_offset(ch1,1)
        self.awg4.set_channel_offset(input_mzi,4)

    def upload_waveforms(self,channels,markers):
        #Channels is a list of length 16 of waveforms to upload
        #Each element of the list either contains elements to upload or is None

        #Markers is a list of length 4 containing the length and delay of the trigger pulse to send
        assert len(channels) == 16
        assert len(markers) == 4

        prescaler_1GSa = 0
        prescaler_200MSa = 1
        prescaler_50MSa = 2

        delay = 0
        cycles = 1
        ext_trigger = keysightSD1.SD_TriggerModes.EXTTRIG
        auto_trigger = keysightSD1.SD_TriggerModes.AUTOTRIG
        software_trigger = keysightSD1.SD_TriggerModes.SWHVITRIG
        
        def upload_to_awg(awg,channel,module_id,prescaler=prescaler_50MSa):        
            if channel is None:
                pass
            else:
                for index,val in enumerate(channel):
                    ref = awg.upload_waveform(val)
                    self.collection_of_waveform_references.append(ref)
                    if index == 0:
                        if self.cyclic == True:
                            awg.awg_queue_waveform(module_id, ref, auto_trigger,  delay, cycles, prescaler)
                        else:
                            awg.awg_queue_waveform(module_id, ref, ext_trigger,  delay, cycles, prescaler)
                    else:
                        awg.awg_queue_waveform(module_id, ref, auto_trigger, delay, cycles, prescaler)

        #Module 1
        upload_to_awg(self.awg1,channels[0],module_id=1)
        upload_to_awg(self.awg1,channels[1],module_id=2)
        upload_to_awg(self.awg1,channels[2],module_id=3)
        upload_to_awg(self.awg1,channels[3],module_id=4)

        #Module 2
        upload_to_awg(self.awg2,channels[4],module_id=1)
        upload_to_awg(self.awg2,channels[5],module_id=2)
        upload_to_awg(self.awg2,channels[6],module_id=3)
        upload_to_awg(self.awg2,channels[7],module_id=4)

        #Module 3
        upload_to_awg(self.awg3,channels[8],module_id=1)
        upload_to_awg(self.awg3,channels[9],module_id=2)
        upload_to_awg(self.awg3,channels[10],module_id=3)
        upload_to_awg(self.awg3,channels[11],module_id=4)

        #Module 4
        upload_to_awg(self.awg4,channels[12],module_id=1)
        upload_to_awg(self.awg4,channels[13],module_id=2)
        upload_to_awg(self.awg4,channels[14],module_id=3)
        upload_to_awg(self.awg4,channels[15],module_id=4)

        def marker_setup(awg,marker,marker_value = 1):
            #Marker object is length, delay
            if marker is None:
                pass
            else:
                awg.set_marker_config(channel_number=1, markerMode=2,trgPXImask=0b00,
                    trgIOmask=0b1, markerValue=marker_value, syncMode=0 ,length=marker[0] , delay=marker[1], verbose = True)

        marker_setup(self.awg1,markers[0],marker_value=1) #IVC102 integrator is default low (resetting) and pulses high for integration
        marker_setup(self.awg2,markers[1],marker_value=1) #Low pulse high
        marker_setup(self.awg3,markers[2],marker_value=1) #Low pulse high
        marker_setup(self.awg4,markers[3],marker_value=1) #Low pulse high

    def run_awg(self):
        self.awg1.awg_start_multiple(0b1111)
        self.awg2.awg_start_multiple(0b1111)
        self.awg3.awg_start_multiple(0b1111)
        self.awg4.awg_start_multiple(0b1111)

        #Send in external trigger to start running time signals
        pxi1 = keysightSD1.SD_TriggerExternalSources.TRIGGER_PXI1
        self.awg1.set_pxi_trigger(0, pxi1-4000)
        self.awg1.set_pxi_trigger(1, pxi1-4000)

    def clear_memory(self):
        #Clears the AWG memory to allow more waveforms to be uploaded
        #First, takes the collection of waveform references and releases them/tells the awg it's okay to erase them
        for i in self.collection_of_waveform_references:
            i.release()
        self.collection_of_waveform_references = []
        #Then instruct each of the AWG channels to flush all of their memory.
        for i in range(1,5):
            self.awg1.awg_flush(i)
            self.awg2.awg_flush(i)
            self.awg3.awg_flush(i)
            self.awg4.awg_flush(i)