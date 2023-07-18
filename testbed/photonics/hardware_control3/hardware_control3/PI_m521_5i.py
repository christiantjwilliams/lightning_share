"""
Python wrapper for PI M521.5i 8" IntelliStage.
Written by C. Panuski <cpanuski@mit.edu>, July 2019

See manual in hardware control folder. This wrapper simply helps with the 
 serial commands.
"""

import numpy as np
import serial
import serial.tools.list_ports as portList

eol = '\n'
maxVelocity = 100000

class PI_m521(object):    
    def __init__(self, port='COM7',timeout=.1):
        # Connect to stage and notify on success
        ports = [aport.device.encode() for aport in portList.comports()]
        if port in ports:
            print('here')
            self.stagePort = port
            self.comms = serial.Serial(port,timeout=timeout)
        else:
            for openPort in ports:
                self.stagePort = openPort
                self.comms = serial.Serial(openPort,timeout=timeout)
                echoFlag = self.sendCommand('test')
                if echoFlag == 0:
                    continue
                else:
                    print((openPort, ' is the wrong port!'))
                    break
        
        # Initialize variables and get status and positions
        self.status = {}
        self.position = {}
        self.velocity = {}
        self.getStatus()
        print(('Statuses are: '+str(self.status)))
        self.getPosition()
        print(('Positions are: '+str(self.position)))
        
#        # Start buffer clearing thread
#        self.clearBuffers()
            
        # Reset stage
        self.reset()
        
#    def clearBuffers(self,Treset=0.1):
#        threading.Timer(Treset, self.clearBuffers).start()
#        self.comms.reset_input_buffer()
#        self.comms.reset_output_buffer()
##        print('Buffers cleared')
            
    def sendCommand(self,command):
        sentChars = self.comms.write((command+eol).encode())
        returnVal = self.comms.read(sentChars)
        if returnVal != (command+eol).encode():
            print('Command error')
            return 1
        else:
            return 0
        
    def sendRequest(self,request):
        self.comms.write((request+eol).encode())
        # Read the echoed command
        self.comms.readline()
        # Read controller response; convert to int
        return int((self.comms.readline()).decode().replace(request.lower()+':',''))
        
    def getStatus(self, axis = np.arange(1,10,1)):
        # If the status of each axis hasn't been initialized, do that; otherwise
        # just check status of active ones.
        if len(list(self.status.keys())):
            axis = list(self.status.keys())
            
        for cmdAxis in axis:
            try:
                self.status[cmdAxis] = ('{0:08b}'.format(self.sendRequest(str(cmdAxis)+'TS')))
            except:
                print(('Axis ' + str(cmdAxis) + ' not found'))
        return
    
    def getPosition(self, axis = None):
        # Get position of the active axes
        if axis == None: #Multi axis
            axis = list(self.status.keys())
            for cmdAxis in axis:
                self.position[cmdAxis] = int(self.sendRequest(str(cmdAxis)+'TP'))
        elif isinstance(axis,int): #Single axis
            self.position[axis] = int(self.sendRequest(str(axis)+'TP'))
            return self.position[axis]
        
    def getVelocity(self, axis = None):
        # Get position of the active axes
        if axis == None: #Multi axis
            axis = list(self.status.keys())
            for cmdAxis in axis:
                self.velocity[cmdAxis] = int(self.sendRequest(str(cmdAxis)+'TY'))
        elif isinstance(axis,int): #Single axis
            self.velocity[axis] = int(self.sendRequest(str(axis)+'TY'))
            return self.velocity[axis]
    
    def moveAbsolute(self, axis, steps, maxRange=False):
        if maxRange:
            steps = np.sign(steps)*2000000 #Max travel range
        self.sendCommand(str(axis)+'MA'+str(int(steps))) 
    
    def moveRelative(self, axis, steps):
        # Reading position is only accurate if stage is stationary
        # So just return if the stage is moving. 
        self.getStatus(axis)
        if int(self.status[axis][7]) == 1: #stage is moving
            print('Error: stage currently moving.')
        else:
            currPos = self.getPosition(axis)
            nextPos = currPos + steps
            self.moveAbsolute(axis,nextPos)
            
    def setVelocity(self, axis, vel):
        if vel >= 1 and vel < 200000:
            self.sendCommand(str(axis)+'SV'+str(int(vel))) 
            print(('Stage ' + str(axis) + ' speed set to '+str(vel)))
        else:
            print(('Stage ' + str(axis) + ' velocity at limit!'))

    def setHome(self, axis = None):
        # Set (0,0) location
        if axis == None: #Multi axis
            self.sendCommand('DH')
        elif isinstance(axis,int): #Single axis
            self.sendCommand(str(axis)+'DH')
        return

    def goHome(self, axis=['']):
        for cmdAxis in axis:
            self.sendCommand(str(cmdAxis)+'MA0') 
        print('Stage homed.')
        
    def abortMotion(self, axis):
        self.sendCommand(str(axis)+'AB') 

    def stopMotion(self):
        self.sendCommand('ST') 
        
    def reset(self,axis=['']):
        for cmdAxis in axis:
            self.sendCommand(str(cmdAxis)+'RT') 
        print('PI M521-5i system reset.')
        
    def shutdown(self):
        self.comms.close()
        print('PI M521-5i shut down.')
    
if __name__ == '__main__':
    # Initialize stage, move to home and shut down comms.
    stage = PI_m521()
#    stage.goHome()
    stage.shutdown()
    print('done')

