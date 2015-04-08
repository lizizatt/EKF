import math
import numpy as np

class Voltmeter():
	def __init__(self,noiseMean, noiseSTD):
		self.noiseMean = noiseMean
		self.noiseSTD = noiseSTD
		self.prevData = 10.0;

	def getData(self):
		"""
		state transfer function xi = (xi-1)^1.01
		"""
		noise = np.random.normal(self.noiseMean,self.noiseSTD,1)
		self.prevData = math.pow(self.prevData,1.01); #state transfer is xi = (xi-1)^1.01 
		if(self.prevData > 100):
			self.prevData = 10.0;
		return self.prevData + noise[0]


class Ammeter():
	def __init__(self, noiseMean, noiseSTD):
		self.noiseMean = noiseMean
		self.noiseSTD = noiseSTD
		self.prevData = 250;

	def getData(self):
		"""
		state transfer function xi = (xi-1)^.99 + 5
		"""
		noise = np.random.normal(self.noiseMean,self.noiseSTD,1)
		self.prevData = math.pow(self.prevData,.99)+5;  
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

