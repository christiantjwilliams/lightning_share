import ctypes
import os
import sys
import numpy as np

class MovingError(AssertionError): pass

class PI843StageController(object):
	"""
	Python wrapper for PI843 Stage Controller.
	Written by Darius Bunandar <dariusb@mit.edu>, July 2015

	Requires the file C843_GCS_DLL.dll to be in the same directory as this file
	The PI C843 controller can control 4 axes. But, sometimes it is used to
	only control 2 axes.
	To control 4 axes, pass a list of 4 stage names to the class.
	To control only 2 axes, pass a list of 2 stage names to the class.
	"""
	c843DLLFile = os.getcwd()+"\\C843_GCS_DLL.dll"
	c843DLL = ctypes.windll.LoadLibrary(c843DLLFile)
	__bufferLen = 2048
	__iMaxLen = ctypes.c_int(__bufferLen)
	__lMaxLen = ctypes.c_long(__bufferLen)

	def __init__(self,listOfStageNames,iBoardNumber=None):
		# Some pointers and variables for IO
		self.__lBoardID = ctypes.c_long()
		self.__cpBuffer = ctypes.c_char_p(' ')

		# Some useful parameters for the stage. These are default parameters,
		# which you can change using the appropriate functions
		defaultVelocity = 25
		defaultAcceleration = 50
		defaultDeceleration = 50

		# If you don't know what the board number is, then the program will
		# exit printing the board number
		if iBoardNumber is None:
			print(self.GetBoardNumber())
			sys.exit()

		"""
		According to the documentation, before we can start moving, we need to:
		1. Connect to the controller board, with function: __Connect
		2. Assign axes to stages, with function: AssignAxesToStages
		3. Initialize the chips, with function: InitializeChips
		4. Set reference modes: SetReferenceModes
		"""
		# Connect to the controller board
		self.__Connect(iBoardNumber)
		self.name = self.GetIdentification()
		# Assign each stage to an axis in the controller board
		# prepare szAxes:
		self.numOfAxes = len(listOfStageNames)
		_ = (str(i+1) for i in range(self.numOfAxes))
		self.__cpSzAxes = ctypes.c_char_p(''.join(_))
		# prepare stage names:
		self.__cpNames = ctypes.c_char_p('\n'.join(listOfStageNames))
		# assigning now:
		self.AssignAxesToStages()

		# Initialize the chip
		self.InitializeChips()

		# By default we set referencing to be off
		refModes = [False for i in range(self.numOfAxes)]
		self.SetReferenceModes(refModes)

		# By default we go to closed-loop mode
		loopModes = [1 for i in range(self.numOfAxes)]
		self.SetLoopModes(loopModes)

		# Set the DEFAULT speed settings
		defaultVelList = [defaultVelocity for i in range(self.numOfAxes)]
		self.SetVelocities(defaultVelList)
		defaultAccList = [defaultAcceleration for i in range(self.numOfAxes)]
		self.SetAccelerations(defaultAccList)
		defaultDecList = [defaultDeceleration for i in range(self.numOfAxes)]
		self.SetDecelerations(defaultDecList)

		# Get the maximum and minimum values
		self.listOfMaxValues = self.GetMaximumValues()
		self.listOfMinValues = self.GetMinimumValues()

	def __Connect(self,iBoardNumber):
		self.__lBoardID \
			= self.c843DLL.C843_Connect(ctypes.c_long(iBoardNumber))
		if self.__lBoardID<0:
			raise ValueError('Wrong board number, cannot connect')

	# def __CheckError(self):
	# 	lErrID = self.c843DLL.C843_GetError(self.__lBoardID)
	# 	success = self.c843DLL.C843_TranslateError(lErrID,\
	# 		self.__cpBuffer,self.__iMaxLen)
	# 	if success:
	# 		print self.__cpBuffer.value
	# 	else:
	# 		raise OverflowError('Buffer size too small')

	def GetBoardNumber(self):
		success = self.c843DLL.C843_ListPCI(self.__cpBuffer,\
			self.__lMaxLen)
		if success:
			return self.__cpBuffer.value # most likely you are going to get 1
		else:
			raise WindowsError('No board detected')

	def IsConnected(self):
		return bool(self.c843DLL.C843_IsConnected(ctypes.c_long(\
														self.__lBoardID)))
	def IsMoving(self):
		bVal = (ctypes.c_bool*self.numOfAxes)()
		pbVal = ctypes.cast(bVal,ctypes.POINTER(ctypes.c_bool))
		success = self.c843DLL.C843_IsMoving(self.__lBoardID,self.__cpSzAxes,\
			pbVal)
		if success:
			isAxesMoving = [bool(pbVal[i]) for i in range(self.numOfAxes)]
			isAnyMoving = True in isAxesMoving # is any axis moving?
			return isAnyMoving
		else:
			raise WindowsError('Failed to obtain moving status')

	#### INITIALIZATION FUNCTIONS ####
	def AssignAxesToStages(self):
		# assigns axis to the stage
		success = self.c843DLL.C843_CST(self.__lBoardID,self.__cpSzAxes,\
			self.__cpNames)
		if not success:
			raise WindowsError('Failed to assign axis to stage')

	def InitializeChips(self):
		# initializes motion control chips
		success = self.c843DLL.C843_INI(self.__lBoardID,self.__cpSzAxes)
		if not success:
			raise WindowsError('Failed to initialize the control chips')

	def SetReferenceModes(self,listOfRefModes):
		# sets reference mode for the axes.
		# True=referencing on, False=referencing off
		# Use referencing off unless you know what you are doing
		if len(listOfRefModes) != self.numOfAxes:
			raise IndexError('length of the list must match number of axes')
		bModes = (ctypes.c_bool*self.numOfAxes)(*listOfRefModes)
		pbModes = ctypes.cast(bModes,ctypes.POINTER(ctypes.c_bool))
		success = self.c843DLL.C843_RON(self.__lBoardID,self.__cpSzAxes,\
			pbModes)
		if not success:
			raise WindowsError('Failed to set the referencing modes')

	def SetLoopModes(self,listOfLoopModes):
		# sets the loop modes for the axes.
		# True=closed loop, False=open loop
		# Use closed loop unless you know what you are doing
		if len(listOfLoopModes) != self.numOfAxes:
			raise IndexError('length of the list must match number of axes')
		# although the documentation says bool array--didn't work
		# seems that int32 works.
		iModes = (ctypes.c_int32*self.numOfAxes)(*listOfLoopModes)
		piModes = ctypes.cast(iModes,ctypes.POINTER(ctypes.c_int32))
		success = self.c843DLL.C843_SVO(self.__lBoardID,self.__cpSzAxes,\
			piModes)
		if not success:
			raise WindowsError('Failed to set the loop modes')

	def SetAbsolutePositions(self,listOfPositions):
		# sets current positions as some new absolute positions
		# units are in mm, accuracy to 0.1um
		if len(listOfPositions) != self.numOfAxes:
			raise IndexError('length of the list must match number of axes')
		dPos = (ctypes.c_double*self.numOfAxes)(*listOfPositions)
		pdPos = ctypes.cast(dPos,ctypes.POINTER(ctypes.c_double))
		success = self.c843DLL.C843_POS(self.__lBoardID,self.__cpSzAxes,pdPos)
		if not success:
			raise WindowsError('Failed to set position')

	def SetVelocities(self,listOfVelocities):
		# sets the velocities for the active axes
		if len(listOfVelocities) != self.numOfAxes:
			raise IndexError('length of the list must match number of axes')
		dVel = (ctypes.c_double*self.numOfAxes)(*listOfVelocities)
		pdVel = ctypes.cast(dVel,ctypes.POINTER(ctypes.c_double))
		success = self.c843DLL.C843_VEL(self.__lBoardID,self.__cpSzAxes,pdVel)
		if not success:
			raise WindowsError('Failed to set velocity')

	def SetAccelerations(self,listOfAccelerations):
		# sets the accelerations for the active axes
		if len(listOfAccelerations) != self.numOfAxes:
			raise IndexError('length of the list must match number of axes')
		dAcc = (ctypes.c_double*self.numOfAxes)(*listOfAccelerations)
		pdAcc = ctypes.cast(dAcc,ctypes.POINTER(ctypes.c_double))
		success = self.c843DLL.C843_ACC(self.__lBoardID,self.__cpSzAxes,pdAcc)
		if not success:
			raise WindowsError('Failed to set acceleration')

	def SetDecelerations(self,listOfDecelerations):
		# sets the decelerations for the active axes
		if len(listOfDecelerations) != self.numOfAxes:
			raise IndexError('length of the list must match number of axes')
		dDec = (ctypes.c_double*self.numOfAxes)(*listOfDecelerations)
		pdDec = ctypes.cast(dDec,ctypes.POINTER(ctypes.c_double))
		success = self.c843DLL.C843_DEC(self.__lBoardID,self.__cpSzAxes,pdDec)
		if not success:
			raise WindowsError('Failed to set acceleration')

	#### MOVING FUNCTIONS ####
	# WARNING: THESE FUNCTIONS ARE UNTESTED (DUE TO EXPT CONSTRAINTS)
	# TEST THESE FIRST BEFORE PROCEEDING
	def MoveInALinearPath(self,listOfNewAbsolutePositions):
		# move to new absolute positions in a linear path
		# units are in mm, accuracy to 0.1um
		if len(listOfNewAbsolutePositions) != self.numOfAxes:
			raise IndexError('length of the list must match number of axes')
		dPos = (ctypes.c_double*self.numOfAxes)(*listOfNewAbsolutePositions)
		pdPos = ctypes.cast(dPos,ctypes.POINTER(ctypes.c_double))
		success = self.c843DLL.C843_MVE(self.__lBoardID,self.__cpSzAxes,pdPos)
		if not success:
			raise WindowsError('Failed to move')

	def MoveRelatively(self,listOfRelativePositions):
		# move to some new positions relative to the current positions
		# We require the stage to stop moving before this command is issued
		# units are in mm, accuracy to 0.1um
		if len(listOfRelativePositions) != self.numOfAxes:
			raise IndexError('length of the list must match number of axes')
		dRelPos = (ctypes.c_double*self.numOfAxes)(*listOfRelativePositions)
		pdRelPos = ctypes.cast(dRelPos,ctypes.POINTER(ctypes.c_double))
		if not self.IsMoving:
			success = self.c843DLL.C843_MVR(self.__lBoardID,self.__cpSzAxes,\
				pdRelPos)
			if not success:
				raise WindowsError('Failed to move relatively')
		else:
			raise MovingError('Stage is still moving')

	def Move(self,listOfNewAbsolutePositions):
		# move to new absolute positions in a NON-LINEAR PATH
		# use MoveInALinearPath instead, if possible
		# units are in mm, accuracy to 0.1um
		if len(listOfNewAbsolutePositions) != self.numOfAxes:
			raise IndexError('length of the list must match number of axes')
		dPos = (ctypes.c_double*self.numOfAxes)(*listOfNewAbsolutePositions)
		pdPos = ctypes.cast(dPos,ctypes.POINTER(ctypes.c_double))
		success = self.c843DLL.C843_MOV(self.__lBoardID,self.__cpSzAxes,pdPos)
		if not success:
			raise WindowsError('Failed to move')

	#### QUERY FUNCTIONS ####
	def GetIdentification(self):
		# get name of controller
		cBuffer = ctypes.c_char()
		success = self.c843DLL.C843_qIDN(self.__lBoardID,\
			self.__cpBuffer,self.__lMaxLen)
		if success:
			return str(self.__cpBuffer.value)
		else:
			raise WindowsError('Failed to retrieve identification')

	def GetNumberOfDataRecorderTables(self):
		# get the number of data recorder tables
		lNrOfTables = ctypes.c_long()
		success = self.c843DLL.C843_qTNR(self.__lBoardID,\
			ctypes.byref(lNrOfTables))
		if success:
			return int(lNrOfTables.value)
		else:
			raise WindowsError('Failed to retrieve data recorder tables')

	def GetConnectedAxes(self):
		# get connected axes numbers
		success = self.c843DLL.C843_qSAI(self.__lBoardID,\
			self.__cpBuffer,self.__lMaxLen)
		if success:
			return str(self.__cpBuffer.value)
		else:
			raise WindowsError('Failed to retrieve connected axes')

	def GetAvailableAxes(self):
		# get the available axes numbers
		success = self.c843DLL.C843_qSAI_ALL(self.__lBoardID,\
			self.__cpBuffer,self.__lMaxLen)
		if success:
			return str(self.__cpBuffer.value)
		else:
			raise WindowsError('Failed to retrieve available axes')

	def GetAvailableStageNames(self):
		# get the available stage names
		pAxes = ctypes.c_char_p("")
		success = self.c843DLL.C843_qCST(self.__lBoardID,pAxes,
			self.__cpBuffer,self.__lMaxLen)
		if success:
			return str(self.__cpBuffer.value)
		else:
			raise WindowsError('Failed to retreive axes')

	def GetPositions(self):
		# get the position of the axes
		dVal = (ctypes.c_double*self.numOfAxes)()
		pdVal = ctypes.cast(dVal,ctypes.POINTER(ctypes.c_double))
		success = self.c843DLL.C843_qPOS(self.__lBoardID,self.__cpSzAxes,\
			pdVal)
		if success:
			listOfPositions = [float(pdVal[i]) for i in range(self.numOfAxes)]
			return listOfPositions
		else:
			raise WindowsError('Cannot get position')

	def GetMinimumValues(self):
		# get minimum values of the axes
		# units are in mm, accuracy to 0.1um
		dValMin = (ctypes.c_double*self.numOfAxes)()
		pdValMin = ctypes.cast(dValMin,ctypes.POINTER(ctypes.c_double))
		success = self.c843DLL.C843_qTMN(self.__lBoardID,self.__cpSzAxes,\
			pdValMin)
		if success:
			listOfMinValues = \
				[float(pdValMin[i]) for i in range(self.numOfAxes)]
			return listOfMinValues
		else:
			raise WindowsError('Failed to obtain minimum value')

	def GetMaximumValues(self):
		# get maximum values of the axes
		# units are in mm, accuracy to 0.1um
		dValMax = (ctypes.c_double*self.numOfAxes)()
		pdValMax = ctypes.cast(dValMax,ctypes.POINTER(ctypes.c_double))
		success = self.c843DLL.C843_qTMX(self.__lBoardID,self.__cpSzAxes,\
			pdValMax)
		if success:
			listOfMaxValues = \
				[float(pdValMax[i]) for i in range(self.numOfAxes)]
			return listOfMaxValues
		else:
			raise WindowsError('Failed to obtain maximum value')

if __name__ == '__main__':
	import time
	start_time = time.time()

	stageNames = ['m-505.4pd','m-505.6pd']
	stage = PI843StageController(stageNames,iBoardNumber=1)
	print((stage.IsConnected()))
	print((stage.GetConnectedAxes()))
	print((stage.GetPositions()))

	print(("--- %s seconds ---" % (time.time() - start_time)))
