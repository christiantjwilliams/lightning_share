import numpy as np
import matplotlib.pyplot as plt
import time

class Methods(object):

	def __init__(self):
		pass

	def FindMaximaAndMinima(self,x,y,delta):

		"""FindMaximaAndMinima
		Given an array of x and y, return a list of indices of
		local maxima and minima.
		Args:	x, y are numpy arrays.
				delta is some threshold value at which we consider a value to be a
				maximum or minimum.
		Return: localMaxIndices, localMinIndices

		To obtain the x and y values of the maxima and minima, e.g.:
			valuesOfLocalMaxima = y[localMaxIndices]
			positionsOfLocalMaxima = x[localMaxIndices]
		Note that this algorithm is O(N) where N = len(x) = len(y).
		"""

		if len(x) != len(y):
			raise IndexError("x and y have do not have matching lengths")

		localMin = float("inf")
		localMax = float("-inf")

		localMaxIndices = []
		localMinIndices = []

		lookForMax = True
		for i in range(len(y)):
			if y[i] > localMax:
				localMax = y[i]
				localMaxPos = x[i]
				localMaxIndex = i
			if y[i] < localMin:
				localMin = y[i]
				localMinPos = x[i]
				localMinIndex = i

			if lookForMax:
				if y[i] < localMax-delta:
					localMaxIndices.append(localMaxIndex)
					localMin = y[i]
					localMinPos = x[i]
					lookForMax = False
			else:
				if y[i] > localMin+delta:
					localMinIndices.append(localMinIndex)
					localMax = y[i]
					localMaxPos = x[i]
					lookForMax = True

		return localMaxIndices, localMinIndices

	def __MeasurePower(self,laser,pm,channel,x):
		laser.set_wavelength(x)
		pm.set_wavelength(x,channel=channel)
		return pm.read_power(channel=channel)

	def AdaptiveSpectrumSweep(self,laser,pm,xCoarse,dyThreshold=3,channel=1,\
		dxMin=None, extremumDelta=3,peaks='all',plot=False):
		"""AdaptiveSpectrumSweep
		For simplicity, x = wavelength (in nm), y = power (in dBm).
		This function performs an adaptive sweep of spectrum by:
		1. 	Minimizing the value of dy between adjacent x points
		2. 	If dxMin is not None, then we are also scanning for peaks up to
			the precision

		Note: This sweep only works for a stable spectrum.
		If your spectrum varies with time, you may be stuck in an infinite loop.
		dyThreshold should be 3 dB or above, or the sweep may take forever.

		Args:
			* laser = handle for SantecTSL710
			* pm = handle for Agilent 8153A powermeter
			* xCoarse = numpy array of the coarse wavelength points.
				Resolution of ~0.1 or 0.2 nm is recommended.
				If you have an idea where the peaks are going to be,
				choose to have coarser resolution at wavelength points that do
				not need the higher resolution.
			* dyThreshold = maximum tolerable dyThreshold (in dB)
			* channel = channel of pm we are measuring index
			* dxMin = wavelength resolution of the peaks (in nm)
			* extremumDelta = how many dB should a peak should be higher
			* peaks = which peaks to scan: 'positive', 'negative', or 'all'
			* plot = set as True for interactive plotting
		Return:
			* x = numpy array of wavelength points
			* y = numpy array of
		"""

		x = xCoarse
		y = np.empty_like(xCoarse)

		if plot:
			plt.ion()
			plt.show()
			plt.xlabel('wavelength (nm)')
			plt.ylabel('power (dBm)')
			ax = plt.gca()
			ax.ticklabel_format(useOffset=False)
			plt.hold(True)

		# initial scan:
		for i in range(len(x)):
			y[i] = self.__MeasurePower(laser,pm,channel,x[i])
			if plot:
				plt.plot(x[:i],y[:i],linestyle='-',marker='x',color='k')
				plt.draw()

		# populate while sparse, breaking when dy is within threshold
		while True:
			dy = np.diff(y)
			xMean = .5*np.diff(x) + x[:(len(x)-1)]

			aboveThreshold = np.where(abs(dy)>dyThreshold)
			if np.size(aboveThreshold[0]) is 0: # no dy is above threshold
				break

			xNew = xMean[aboveThreshold]
			yNew = np.empty_like(xNew)
			for i in range(len(xNew)):
				yNew[i] = self.__MeasurePower(laser,pm,channel,xNew[i])
				if plot:
					plt.scatter(xNew[i],yNew[i],marker='o',color='r')
					plt.draw()

			x = np.append(x,xNew)
			y = np.append(y,yNew)
			p = x.argsort()
			x = x[p]
			y = y[p]

			if plot:
				plt.clf()
				plt.plot(x,y,linestyle='-',marker='x',color='k')
				plt.draw()

		# try to resolve the peaks within uncertainty of dxMin
		if dxMin is not None:
			localMaxIndices, localMinIndices = FindMaximaAndMinima(x,y,extremumDelta)

			if peaks == 'positive':
				index = localMaxIndices
			elif peaks == 'negative':
				index = localMinIndices
			elif peaks == 'all':
				index = localMaxIndices + localMinIndices
			else:
				raise ValueError("Invalid choice")

			xNew = []
			yNew = []

			for i in index:
				if i is not 0:
					currentX = x[i]-dxMin
					while currentX >= (x[i-1]+dxMin):
						currentY = self.__MeasurePower(laser,pm,channel,currentX)
						xNew.append(currentX)
						yNew.append(currentY)
						if plot:
							plt.scatter(currentX,currentY,marker='o',color='r')
							plt.draw()
						currentX -= dxMin

				if i is not len(x):
					currentX = x[i]+dxMin
					while currentX <= (x[i+1]-dxMin):
						currentY = self.__MeasurePower(laser,pm,channel,currentX)
						xNew.append(currentX)
						yNew.append(currentY)
						if plot:
							plt.scatter(currentX,currentY,marker='o',color='r')
							plt.draw()
						currentX += dxMin

			x = np.append(x,xNew)
			y = np.append(y,yNew)
			p = x.argsort()
			x = x[p]
			y = y[p]

		return x, y
