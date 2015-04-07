import math
import numpy as np


resistor = 50 #ohms

class Voltmeter():
	def __init__(self,noiseMean, noiseSTD):
		self.noiseMean = noiseMean
		self.noiseSTD = noiseSTD
		self.function = math.sin

	def getData(self,time):
		noise = np.random.normal(self.noiseMean,self.noiseSTD,1)
		return self.function(time/3.14) + noise[0]


class Ammeter():
	def __init__(self, noiseMean, noiseSTD):
		self.noiseMean = noiseMean
		self.noiseSTD = noiseSTD
		self.function = math.cos

	def getData(self,time):
		noise = np.random.normal(self.noiseMean,self.noiseSTD,1)
		return self.function(time/3.14) + noise[0]





class Voltmeter_Linear():
	def __init__(self,noiseMean, noiseSTD):
		self.noiseMean = noiseMean
		self.noiseSTD = noiseSTD
		self.function = math.sin
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
		self.function = math.cos
		self.prevData = 12;
		self.scaleFactor = 1.01;

	def getData(self,time):
		noise = np.random.normal(self.noiseMean,self.noiseSTD,1)
		self.prevData = self.prevData * self.scaleFactor;
		return self.prevData + noise[0]