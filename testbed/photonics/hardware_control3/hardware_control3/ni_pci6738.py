import nidaqmx
import numpy as np


class NI_PCIe6738(object):
    def __init__(self, device=None):
        if device is None:
            self.__find_device()
        else:
            self.device = device

    def __find_device(self):
        # Finds the first PCIe-6738 device
        system = nidaqmx.system.System.local()
        for device in system.devices:
            if device.product_type == "PCIe-6738":
                self.device = device.name

    def set_voltage(self, port, voltage):
        # Set the voltage of a single output analog port
        # Voltage can be a row vector for varying values
        with nidaqmx.Task() as task:
            task.ao_channels.add_ao_voltage_chan(self.device+"/ao"+str(port))
            task.write(voltage, auto_start=True)

    def set_multiple_voltages_continuous(self, min_port, max_port, voltages):
        # Set the voltages of multiple output analog ports defined by 
        # min number and max number
        # Voltages can be a single column vector 
        # (whose length is the number of ports)
        # or can be a matrix, each row corresponds to one port
        with nidaqmx.Task() as task:
            task.ao_channels.add_ao_voltage_chan(
                self.device+"/ao"+str(min_port)+":"+str(max_port))
            task.write(voltages, auto_start=True)

    def set_multiple_voltages(self, ports, voltages):
        # Set the voltages of multiple output analog ports
        # Voltages can be a single column vector 
        # (whose length is the number of ports)
        # or can be a matrix, each row corresponds to one port
        if len(ports) != len(voltages):
            raise ValueError("Number of voltages and ports don't match")
        with nidaqmx.Task() as task:
            for port in ports:
                task.ao_channels.add_ao_voltage_chan(
                    self.device+"/ao"+str(port))
            task.write(voltages, auto_start=True)

    def zero_all_voltages(self):
        self.set_multiple_voltages_continuous(0, 31, np.zeros([32, 1]))

    def __del__(self):
        self.zero_all_voltages()  # to be safe set all zeros


if __name__ == '__main__':
    import time
    analog_device = NI_PCIe6738()

    voltages = .1*np.array(list(range(1, 33)))
    voltages = np.reshape(voltages, [32, 1])
    print("setting continuous array of voltages")
    analog_device.set_multiple_voltages_continuous(0, 31, voltages)
    time.sleep(10)
    
    print("setting everything back to zero")
    analog_device.zero_all_voltages()

    print("setting a few voltages")
    ports = [0, 15, 31]
    voltages = np.array([[-.5], [-1], [-2]])
    analog_device.set_multiple_voltages(ports, voltages)
    time.sleep(10)
    print("done")
