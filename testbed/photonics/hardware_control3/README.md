# hardware_control3
Python 3 software for controlling lab hardware. The following hardware is currently supported,
* Agilent E3631
* Agilent8153A
* Agilent 86141B
* Agilent DS081004A
* Anritsu MG9638A
* Anritsu MP1763B
* Arroyo 5240
* EXFO FVA 3100
* HP 8720B
* JDSU HA9
* Keithley 2400
* Keithley 2450
* Newport 1936/2936
* Santec TSL210
* Santec TSL710
* Tektronix AFG3252
* Tektronix 5014C
* Thorlabs MDT Piezo
* PI C843 stage
* SRS SIM 900 main frame with SIM 928 voltage sources
* Thorlabs Z825B and other APT devices

## Installation
* First, take a minute to learn [git](http://rogerdudler.github.io/git-guide/)
* Via Python's pip package manager
	* Copy the SSH address for the git project from github.mit.edu
	* Add a public [SSH key](https://help.github.com/enterprise/2.1/user/articles/generating-ssh-keys) for your computer
	* Run `pip install git+ssh://git@github.mit.edu/saumilb/hardware_control3.git`
* Updating the pip package
	* run `pip install --upgrade hardware_control3`
* Installation without pip
	* Add the git project directory to your `PYTHONPATH` environment variable
		* OS X: (1) modify `~/.bash_profile` to include `export PYTHONPATH=/my/directories/git/hardware_control3:$PYTHONPATH`, (2) run `source ~/.bash_profile`. For example, see [stackoverflow] (http://stackoverflow.com/questions/3387695/add-to-python-path-mac-os-x)

## Requirements
* Python
  * `pip install pyvisa`
  * `pip install PyThorlabsMDT`
* Software
  * [National Instruments 488.2 Mac](http://www.ni.com/download/ni-488.2-14.0/4802/en/)
  * [National Instruments VISA Mac](http://www.ni.com/download/ni-visa-14.0.2/5075/en/)

## Usage
If you wanted to use the laser, you could import it after installing the package via pip (see above) as so `from hardware_control3.santectsl710 import SantecTSL710 as laser`.

## Adding to the `README.md`
[Markdown Guide](https://guides.github.com/features/mastering-markdown/).
