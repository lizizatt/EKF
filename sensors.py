import math
import numpy as np


resistor = 50 #ohms

class Voltmeter():
	def __init__(self,noiseMean, noiseSTD):
		self.noiseMean = noiseMean
		self.noiseSTD = noiseSTD
		self.prevData = 5.0;

	def getData(self,time):
		"""
		apply arbitrary function
		"""
		noise = np.random.normal(self.noiseMean,self.noiseSTD,1)
		self.prevData = math.pow(self.prevData,1.01); #state transfer is xi = (xi-1)^1.01 
		return self.prevData + noise[0]


class Ammeter():
	def __init__(self, noiseMean, noiseSTD):
		self.noiseMean = noiseMean
		self.noiseSTD = noiseSTD
		self.prevData = 150;

	def getData(self,time):
		"""
		apply arbitrary function
		"""
		noise = np.random.normal(self.noiseMean,self.noiseSTD,1)
		self.prevData = math.pow(self.prevData,.99);  #state transfer is xi = sqrt(xi-1)
		return self.prevData + noise[0]


class Voltmeter_Linear():
	def __init__(self,noiseMean, noiseSTD):
		self.noiseMean = noiseMean
		self.noiseSTD = noiseSTD
		self.prevData = 100;
		self.scaleFactor = .95;

	def getData(self,time):
		noise = np.random.normal(self.noiseMean,self.noiseSTD,1)
		self.prevData = self.prevData * self.scaleFactor;
		return self.prevData + noise[0]


class Ammeter_Linear():
	def __init__(self, noiseMean, noiseSTD):
		self.noiseMean = noiseMean
		self.noiseSTD = noiseSTD
		self.prevData = 12;
		self.scaleFactor = 1.01;

	def getData(self,time):
		noise = np.random.normal(self.noiseMean,self.noiseSTD,1)
		self.prevData = self.prevData * self.scaleFactor;
		return self.prevData + noise[0]