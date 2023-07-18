import numpy as np
import pyAPT
import sys
import time
import pytesseract
import matplotlib.pyplot as plt
import pyscreenshot as ImageGrab
from hardware_control.agilent_8153a import Agilent8153A as pm
from PIL import Image

SER = 83845709
PATH = '/Users/nick/Documents/thorlabs_tdc001/pyAPT-master/'


class ThorlabsZ825B:

    def __init__(self):
        self.serial = 83845709
        self.con = pyAPT.Z825B(serial_number=self.serial)

        # set velocity, acceleration of stage
        self.con.set_velocity_parameters(.3, .3)
        min_vel, acc, max_vel = self.con.velocity_parameters()

    def move(self, x):
        '''
        Move the stage to position (x) in [mm].
        '''

        self.con.goto(float(x), wait=False)

        stat = self.con.status()
        while stat.moving:
            print 'Moving...'
            time.sleep(.01)
            stat = self.con.status()
            print 'At (%.4f) mm with velocity (%.4f) mm/s.' % (stat.position, stat.velocity)

        return self.con.position()

    def home(self):
        '''
        Set the stage to its home position.
        '''

        self.con.home(velocity=.5)

    def reset(self):
        '''
        Resets all controller parameters to their default EEPROM value.
        '''
        self.con.reset_parameters()

    def get_info(self):
        '''
        Gets the controller information of all APT controllers, or the one specified.
        '''
        return self.con.info()

    def __del__(self):
        self.con.close()

if __name__ == '__main__':
    stage = ThorlabsZ825B()
    stage.home()
    print stage.move(11.3)
