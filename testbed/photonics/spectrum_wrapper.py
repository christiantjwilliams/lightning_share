import sys

# import spectrum driver functions
from pyspcm import *
from spcm_tools import *
import numpy as np 

class spectrum():
    def __init__(self):
        self.segment_size = 16
        # self.segment_size = 2048
        self.setup()

    def setup(self):
        szErrorTextBuffer = create_string_buffer(ERRORTEXTLEN)
        dwError = uint32()
        self.lStatus = int32()
        self.lAvailUser = int32()
        self.lPCPos = int32()
        self.qwTotalMem = uint64(0)
        self.qwToTransfer = uint64(MEGA_B(8))


        ############### CONNECT TO SPECTRUM INSTRUMENTS CARD ########################
        self.hCard = spcm_hOpen(create_string_buffer(b'/dev/spcm0'))
        if self.hCard == None:
            sys.stdout.write("no card found...\n")
            exit(1)

        # read type, function and sn and check for A/D card
        lCardType = int32(0)
        spcm_dwGetParam_i32(self.hCard, SPC_PCITYP, byref(lCardType))
        lSerialNumber = int32(0)
        spcm_dwGetParam_i32(self.hCard, SPC_PCISERIALNO, byref(lSerialNumber))
        lFncType = int32(0)
        spcm_dwGetParam_i32(self.hCard, SPC_FNCTYPE, byref(lFncType))

        sCardName = szTypeToName(lCardType.value)
        if lFncType.value == SPCM_TYPE_AI:
            sys.stdout.write("Found: {0} sn {1:05d}\n".format(
                sCardName, lSerialNumber.value))
        else:
            sys.stdout.write(
                "This is an example for A/D cards.\nCard: {0} sn {1:05d} not supported by example\n".format(sCardName, lSerialNumber.value))
            spcm_vClose(self.hCard)
            exit(1)

        # determine the number of channels on the card
        lNumModules = int32(0)
        spcm_dwGetParam_i32(self.hCard, SPC_MIINST_MODULES, byref(lNumModules))
        lNumChPerModule = int32(0)
        spcm_dwGetParam_i32(self.hCard, SPC_MIINST_CHPERMODULE, byref(lNumChPerModule))
        lNumChOnCard = int32(0)
        lNumChOnCard = lNumModules.value * lNumChPerModule.value

        # do a simple standard setup
        # enable all channels on card
        spcm_dwSetParam_i32(self.hCard, SPC_CHENABLE,  1) 
        # spcm_dwSetParam_i32(self.hCard, SPC_CHENABLE,  2)        
        spcm_dwSetParam_i32(self.hCard, SPC_SEGMENTSIZE,  self.segment_size)
        # half of the total number of samples after trigger event
        spcm_dwSetParam_i32(self.hCard, SPC_PRETRIGGER,    self.segment_size//2)
        # half of the total number of samples after trigger event
        spcm_dwSetParam_i32(self.hCard, SPC_POSTTRIGGER,    self.segment_size//2)
        # single trigger standard mode
        spcm_dwSetParam_i32(self.hCard, SPC_CARDMODE,       SPC_REC_STD_MULTI)
        spcm_dwSetParam_i32(self.hCard, SPC_TIMEOUT, 5000)                   # timeout 5 s
        # trigger set to software
        spcm_dwSetParam_i32(self.hCard, SPC_TRIG_EXT0_MODE, SPC_TM_POS)
        spcm_dwSetParam_i32(self.hCard, SPC_TRIG_EXT0_MODE, SPCM_XMODE_TRIGIN)   # set trigger mode
        # trigger set to external
        spcm_dwSetParam_i32(self.hCard, SPC_TRIG_ORMASK,  SPC_TMASK_EXT0)
        spcm_dwSetParam_i32(self.hCard, SPC_TRIG_ANDMASK,  0)                      # ...
        spcm_dwSetParam_i32(self.hCard, SPC_TRIG_EXT0_ACDC, COUPLING_DC)           # trigger coupling

        spcm_dwSetParam_i32(self.hCard, SPC_TRIG_EXT0_LEVEL0,750)  # trigger level of 1.5 Volt
        spcm_dwSetParam_i32(self.hCard, SPC_TRIG_EXT0_LEVEL1,0)  # unused
        # clock mode internal PLL
        spcm_dwSetParam_i32(self.hCard, SPC_CLOCKMODE,      SPC_CM_INTPLL)

        #### Set everything to 50 ohm termination
        #Trigger terminations
        spcm_dwSetParam_i32(self.hCard, SPC_TRIG_TERM,1)
        #Analog input terminations
        spcm_dwSetParam_i32(self.hCard, SPC_50OHM0, 1)
        spcm_dwSetParam_i32(self.hCard, SPC_50OHM1, 1)
        spcm_dwSetParam_i32(self.hCard, SPC_50OHM2, 1)
        spcm_dwSetParam_i32(self.hCard, SPC_50OHM3, 1)
        spcm_dwSetParam_i32(self.hCard, SPC_50OHM4, 1)
        spcm_dwSetParam_i32(self.hCard, SPC_50OHM5, 1)
        spcm_dwSetParam_i32(self.hCard, SPC_50OHM6, 1)
        spcm_dwSetParam_i32(self.hCard, SPC_50OHM7, 1)

        self.lSetChannels = int32(0)
        # get the number of activated channels
        spcm_dwGetParam_i32(self.hCard, SPC_CHCOUNT,     byref(self.lSetChannels))
        # for M2i the data sorting depends on the enabled channels. Here we use a static value for simplicity
        bBothModulesUsed = True

        # setup the channels
        for lChannel in range(0, self.lSetChannels.value, 1):
            # set input range to +/- 200 mV
            spcm_dwSetParam_i64(self.hCard, SPC_AMP0 + lChannel *
                                (SPC_AMP1 - SPC_AMP0),  int32(200))

        # we try to set the samplerate to 100 kHz (M2i) or 20 MHz on internal PLL, no clock output
        spcm_dwSetParam_i64(self.hCard, SPC_SAMPLERATE, MEGA(80))

        # no clock output
        spcm_dwSetParam_i32(self.hCard, SPC_CLOCKOUT, 0)

    def start_triggered_acquisition(self,numPts):
        # settings for the DMA buffer
        # in bytes. Enough memory for 16384 samples with 2 bytes each, all channels active
        qwBufferSize = uint64(numPts* self.segment_size * 2 * self.lSetChannels.value)
        # driver should notify program after all data has been transfered
        lNotifySize = int32(0)

        # define the data buffer
        # we try to use continuous memory if available and big enough
        self.pvBuffer = c_void_p()
        qwContBufLen = uint64(0)
        spcm_dwGetContBuf_i64(self.hCard, SPCM_BUF_DATA, byref(self.pvBuffer), byref(qwContBufLen))
        # sys.stdout.write("ContBuf length: {0:d}\n".format(qwContBufLen.value))
        if qwContBufLen.value >= qwBufferSize.value:
            sys.stdout.write("Using continuous buffer\n")
        else:
            self.pvBuffer = pvAllocMemPageAligned(qwBufferSize.value)
            # sys.stdout.write("Using buffer allocated by user program\n")

        spcm_dwDefTransfer_i64(self.hCard, SPCM_BUF_DATA, SPCM_DIR_CARDTOPC,
                            lNotifySize, self.pvBuffer, uint64(0), qwBufferSize)
        spcm_dwSetParam_i32(self.hCard, SPC_MEMSIZE, self.segment_size*numPts)
        # start card and DMA
        dwError = spcm_dwSetParam_i32(
            self.hCard, SPC_M2CMD, M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER | M2CMD_DATA_STARTDMA)

    def parse_acquisition_data(self,numPts):
        # arrays for minimum and maximum for each channel
        alMin = [32767] * self.lSetChannels.value  # normal python type
        alMax = [-32768] * self.lSetChannels.value  # normal python type

        alSamplePos = []  # array for the position of each channel inside one sample
        
        for lChannel in range(0, self.lSetChannels.value, 1):
            alSamplePos.append(lChannel)

        dwError = spcm_dwSetParam_i32(
            self.hCard, SPC_M2CMD, M2CMD_CARD_WAITREADY | M2CMD_DATA_WAITDMA)
        if dwError != ERR_OK:
            if dwError == ERR_TIMEOUT:
                sys.stdout.write("... Timeout\n")
            else:
                sys.stdout.write("... Error: {0:d}\n".format(dwError))

        else:
            spcm_dwGetParam_i32(self.hCard, SPC_M2STATUS,            byref(self.lStatus))
            spcm_dwGetParam_i32(self.hCard, SPC_DATA_AVAIL_USER_LEN, byref(self.lAvailUser))
            spcm_dwGetParam_i32(self.hCard, SPC_DATA_AVAIL_USER_POS, byref(self.lPCPos))

            self.qwTotalMem.value = self.lAvailUser.value
            # sys.stdout.write("Stat:{0:08x} Pos:{1:08x} Avail:{2:08x} Total:{3:.2f}MB/{4:.2f}MB\n".format(self.lStatus.value, self.lPCPos.value,
                            # self.lAvailUser.value, c_double(self.qwTotalMem.value).value / MEGA_B(1), c_double(self.qwToTransfer.value).value / MEGA_B(1)))

            # this is the point to do anything with the data
            # e.g. calculate minimum and maximum of the acquired data
            lBitsPerSample = int32(0)
            spcm_dwGetParam_i32(self.hCard, SPC_MIINST_BITSPERSAMPLE, byref(lBitsPerSample))
            ch1_data = []
            if lBitsPerSample.value <= 8:
                pass
            else:
                pnData = cast(self.pvBuffer, ptr16)  # cast to pointer to 16bit integer
                for i in range(0, numPts*self.segment_size, 1):
                    for lChannel in range(0, self.lSetChannels.value, 1):
                        lDataPos = i * self.lSetChannels.value + alSamplePos[lChannel]
                        if lChannel == 0:
                            ch1_data.append(pnData[lDataPos])
                        if pnData[lDataPos] < alMin[lChannel]:
                            alMin[lChannel] = pnData[lDataPos]
                        if pnData[lDataPos] > alMax[lChannel]:
                            alMax[lChannel] = pnData[lDataPos]
            truncated_data = [np.mean(ch1_data[16*i+7:16*i+12]) for i in range(len(ch1_data)//16)] 
    
        return truncated_data

    def __del__(self):
        # clean up
        spcm_vClose(self.hCard)