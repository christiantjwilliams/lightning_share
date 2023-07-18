from __future__ import absolute_import, division
from .controller import Controller

class Z825B(Controller):
  """
  A controller for a Z825B stage.
  """
  def __init__(self,*args, **kwargs):
    super(Z825B, self).__init__(*args, **kwargs)

    # http://www.thorlabs.com/thorcat/17600/Z825B-Manual.pdf
    # Note that these values are pulled from the APT User software,
    # as they agree with the real limits of the stage better than
    # what the website or the user manual states
    self.max_velocity = .4 #mm/s
    self.max_acceleration = .4 #mm/s/s

    # from private communication with thorlabs tech support:
    # steps per revolution: 512
    # gearbox ratio: 67
    # pitch: 1 mm
    # thus to advance 1 mm you need to turn 67*512 times
    enccnt = 512.*67
    T = 2048./6e6

    # these equations are taken from the APT protocol manual
    self.position_scale = enccnt
    self.velocity_scale = enccnt * T * (2**16)
    self.acceleration_scale = enccnt * (T**2) * (2**16)

    self.linear_range = (0,25)
