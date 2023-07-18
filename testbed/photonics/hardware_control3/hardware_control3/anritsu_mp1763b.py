import visa
from decimal import Decimal
import math

class InvalidPageError(ValueError): pass

class AnritsuMP1763B(object):

    """
    Python wrapper for AnritsuMP1763B pulse pattern generator (PPG)
    """

    def __init__(self, address, timeout=10000):
        rm = visa.ResourceManager()
        self.instr = rm.open_resource(address)
        self.name = self.instr.query('*IDN?')
        self.__timeout = timeout
        self.instr.timeout = self.__timeout

    def __page_limits(self,firstPage,lastPage,maxPage):
        if(firstPage<1):
            raise InvalidPageError("page<1 doesn't make sense")
        elif(lastPage>maxPage):
            raise InvalidPageError("exceeding maximum number of pages")
        else:
            return firstPage

    def __value_limits(self,val,minVal,maxVal,stepVal):
        notWithinStep = (abs(Decimal(str(val))) % Decimal(str(stepVal))) != 0
        if(val<minVal or val>maxVal or notWithinStep ):
            raise ValueError("improper input value")
        else:
            return val

    def _set_offset_reference(self,ref=2):
        self.instr.write('ofs '+str(ref))

    def _set_clock_resolution(self,res=1):
        self.instr.write('res '+str(res))

    def _fourHexes_to_sixteenBits(self,fourHexes): #with padded zero
        return "{0:016b}".format(int(fourHexes,16))

    def _sixteenBits_to_fourHexes(self,sixteenBits): #with padded zero
        return "{0:04x}".format(int(sixteenBits,2))

    def _read_values(self,command):
        string = self.instr.query(command)
        return string[3:]

    def setup_basic(self):
        self._set_offset_reference()
        self._set_clock_resolution()
        self.set_generation_pattern('data')
        self.set_all(0)
        self.set_data_amplitude(0.25)
        self.set_data_offset(0)
        self.set_notdata_amplitude(0.25)
        self.set_notdata_offset(0)
        self.set_output('on')

    def reset(self):
        self.instr.write('*RST')
        self.instr.timeout = self.instr.__timeout

    def clearall(self):
        self.instr.write('*CLS')
        self.instr.timeout = self.instr.__timeout

    def set_datalength(self,dataLength):
        try:
    	   self.instr.write('dln '+str(self.__value_limits(dataLength,2,8388608,1)))
        except ValueError:
            print("min value=2, max value=8388608")

    def get_datalength(self):
    	val = self._read_values('dln?')
        return int(val)

    def get_total_page_numbers(self):
        dataLength = self.get_datalength()
        return int(math.ceil(float(dataLength)/16))

    def set_frequency(self,frequency): # frequency is set in MHz
        try:
    	   self.instr.write('frq '+str(self.__value_limits(frequency,50,12500,1)))
        except ValueError:
            print('improper input frequency, 50MHz to 12500MHz, step 1MHz')

    def get_frequency(self):
        val = self._read_values('frq?')
        return float(val)

    def get_filenumber(self):
    	val = self._read_values('fil?')
        return int(val)

    def load_file_from_floppy(self,fileNumber):
    	self.instr.write('rcl '+str(fileNumber))

    def save_file_into_floppy(self,fileNumber):
    	self.instr.write('save '+str(fileNumber))

    def delete_file_in_floppy(self,fileNumber):
    	self.instr.write('del '+str(fileNumber))

    def set_generation_pattern(self,selection):
    	# selection is either alternate, data, zero, or PRBS
    	if(selection.lower()=='alternate'):
    		selectionNum = 0
    	elif(selection.lower()=='data'):
    		selectionNum = 1
    	elif(selection.lower()=='zero'):
    		selectionNum = 2
    	elif(selection.lower()=='prbs'):
    		selectionNum = 3
    	else:
    		print('invalid selection')
    		raise
    	self.instr.write('pts '+str(selectionNum))

    def get_generation_pattern(self):
    	selectionNum = self.instr.query('pts?')
    	if(selectionNum==0):
    		return 'alternate'
    	elif(selectionNum==1):
    		return 'data'
    	elif(selectionNum==2):
    		return 'zero'
    	elif(selectionNum==3):
    		return 'prbs'

    def set_output(self,stat='on'):
    	if(stat.lower()=='on'):
    		self.instr.write('oon 1')
    	elif(stat.lower()=='off'):
    		self.instr.write('oon 0')

    def get_output(self):
    	onOrOff=int(self._read_values('oon?'))
        if(onOrOff):
            return 'on'
        else:
            return 'off'

    # page and bit manipulation
    def set_page_number(self,pageNumber): # set starting page number
    	self.instr.write('pag '+str(pageNumber))

    def get_page_number(self): # get starting page number
        val = self._read_values('pag?')
        return int(val)

    def set_eightpage_bits(self,bitsList,startingPage=None): 
        # There are 16 bits per page.
        # This function writes a list of 16 binary bits for each page (8 pages max).
        if startingPage is None:
            startingPage = self.get_page_number()

        try:
            numberOfPagesToBeWritten = len(bitsList)
            numberOfPagesToBeWritten = self.__value_limits(numberOfPagesToBeWritten,1,8,1)
            lastPage = startingPage+len(bitsList)-1
            maxPage = self.get_total_page_numbers()
            startingPage=self.__page_limits(startingPage, lastPage, maxPage)
        except InvalidPageError:
            print("invalid page, check your maximum page")
        except ValueError:
            print("too many pages, only 8 pages at a time")

        fourHexList=[]
        for sixteenBits in bitsList:
            fourHexList.append('#H'+str(self._sixteenBits_to_fourHexes(sixteenBits))+',')
        hexString='PAG '+str(startingPage)+';BIT '+''.join(fourHexList)
        hexString=hexString[:-1]
    	self.instr.write(hexString)

    def get_eightpage_bits(self): 
        # There are 16 bits per page. 
        # This function returns a list of 16 binary bits for each page (8 pages max).
        # To check the starting page use get_page_number
    	raw=self.instr.query('bit?')
        # decode the hex bits
        raw=raw[18:]
        raw=raw.split(',')
        bitsList=[]
        for fourHexes in raw:
            bitsList.append(self._fourHexes_to_sixteenBits(fourHexes[2:]))
        return bitsList

    def set_all(self,bit): # set all to be on (bit=1) or off (bit=0)
    	self.instr.write('all '+str(bit))

    def set_page_to_single_value(self,bit): # set the whole page to be on (bit=1) or off(bit=0)
    	self.instr.write('pst '+str(bit))

    # voltage output levels
    def set_data_offset(self,voltageOffset):
        try:
            self.instr.write('dos '+str(self.__value_limits(voltageOffset,-2,2,.001)))
        except ValueError:
            print("improper data offset, limits -2V to 2V, step 0.001")

    def get_data_offset(self):
        val = self._read_values('dos?')
        return float(val)

    def set_data_amplitude(self,voltageAmplitude):
        try:
            self.instr.write('dap '+str(self.__value_limits(voltageAmplitude,.25,2,.002)))
        except ValueError:
            print("improper data amplitude, limits .25V to 2V, step 0.002")

    def get_data_amplitude(self):
        val = self._read_values('dap?')
        return float(val)

    def set_notdata_offset(self,voltageOffset):
        try:
            self.instr.write('nos '+str(self.__value_limits(voltageOffset,-2,2,.001)))
        except ValueError:
            print("improper notdata offset, limits -2V to 2V, step 0.001")

    def get_notdata_offset(self):
        return float(self.instr.query('nos?')[0])

    def set_notdata_amplitude(self,voltageAmplitude):
        try:
            self.instr.write('nap '+str(self.__value_limits(voltageAmplitude,.25,2,.002)))
        except ValueError:
            print("improper notdata amplitude, limits .25V to 2V, step 0.002")

    def get_notdata_amplitude(self):
        val = self._read_values('nap?')
        return float(val)

if __name__ == '__main__':
    import numpy as np
    ppg = AnritsuMP1763B('GPIB0::8')
    L = 4*100
    ppg.set_datalength(L)
    
    data = []
    for eightpage in range(L/(16*8)):
        N = 8
        bitList = []
        for page in range(N):
            bits = ""
            for i in range(4):
                bit = np.random.randint(2)
                data.append(bit)
                if bit: # 1
                    bits += "0010"
                else: # 0
                    bits += "1000"
            bitList.append(bits[::-1])
        ppg.set_page_number(8*eightpage+1)
        ppg.set_eightpage_bits(bitList, 8*eightpage+1)
    ppg.set_page_number(1)
    data = np.asarray(data)
    print(data)
    a,b = np.unique(data, return_counts=True)
    print(len(data))
    print(b)
    np.savetxt("pattern.txt", data, fmt="%i", delimiter=",")
