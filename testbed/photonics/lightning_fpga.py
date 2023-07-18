import datetime
import os


class LightningFPGA:
    def __init__(self, module):
        self.timecode = "{:%Y-%m-%d_%H%M%S}".format(datetime.datetime.now())

    # ask the FPGA to send the pre-programed calibration waveform
    def send_alibration_waveform():
        pass


class LightningDAC:
    def __init__(self, module):
        pass

    def run_saved_waveforms(saved_waveforms):
        if saved_waveforms == "mod":
            reg_index = 0
            reg_bit = 0
            bit_info = 0
            # send the preloaded waveform through FPGA DAC, controled through XFCP
            os.system("python ../../fpga/lib/xfcp/python/xfcp_ctrl.py --write {} {} {}".format(reg_index, reg_bit, bit_info))  


class LightningADC:
    def __init__(self, module):
        pass

    def start_triggered_acquisition():
        pass
