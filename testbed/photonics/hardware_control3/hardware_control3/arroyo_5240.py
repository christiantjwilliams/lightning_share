import serial
class Arroyo5240(object):
    def __init__(self, port='/dev/tty.usbserial-AL007ZZG'):
      self.ser=serial.Serial(port)
      self.ser.baudrate=38400
      self.ser.bytesize=serial.EIGHTBITS    #these are default values
      self.ser.parity=serial.PARITY_NONE
      self.ser.stopbits=serial.STOPBITS_ONE


    def send_rcv(self, command):
      self.ser.write(command+'\n')
      answer = self.ser.readline()
      return answer.strip()

    def send(self, command):
      self.ser.write(command+'\n')


    def reset(self):
      self.ser.write('*RST\n')


    def decrease_temp(self,steps):
      self.ser.write('TEC:DEC '+str(steps)+'\n')

    def increase_temp(self,steps):
      self.ser.write('TEC:INC '+str(steps)+'\n')

    def get_temp(self):
      self.ser.write('TEC:T?\n')
      temp=self.ser.readline()
      return float(temp.strip())

    def set_temp(self, temp):
      self.ser.write('TEC:T '+str(temp)+'\n')

    def get_set_temp(self):
      self.ser.write('TEC:T?\n')
      set_temp=self.ser.readline()
      return float(set_temp.strip())

    def get_step(self):
      self.ser.write('TEC:TSTEP?\n')
      step=self.ser.readline()
      return float(step.strip())

    def set_step(self,step):     #can't be lower than resolution: 0.01
      self.ser.write('TEC:TSTEP '+str(step)+ '\n')

    def set_temp_limits(self, t_hi,t_lo):
      self.ser.write('TEC:LIM:THI '+str(t_hi)+ '\n')
      self.ser.write('TEC:LIM:TLO '+str(t_lo)+ '\n')

  #PID Autotune--make sure temp high and low limits are pre-set

    def autotune(self, test_temp):
      self.ser.write('TEC:AUTOTUNE '+str(test_temp)+ '\n')

if __name__ == '__main__':
    import time

    obj = Arroyo5240()
    obj.set_temp(49)
    print(obj.get_temp())
    obj.increase_temp(10)
    print(obj.get_temp())
