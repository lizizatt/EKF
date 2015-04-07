import sensors
import math
import numpy as np
import matplotlib.pyplot as plt
import numdifftools as nd




class SimpleKalman():
	"""
	Linear implementation of kalman filter
	"""
	def __init__(self,initialObservation):
		self.estimate = initialObservation  #current
		self.previousEstimate = initialObservation   
		self.gain = np.identity(2) #tradeoff of system between estimation and observation
		self.previousGain = np.identity(2)
		self.errorPrediction = np.identity(2) #current estimation of the signal error,... starts as identity
		self.previousErrorPrediction = np.identity(2)
		self.sensorNoiseProperty = np.array([[math.pow(1,2),0],[0,math.pow(3,2)]])  #variance of sensor noise, ... for us, 0
		self.A = np.array([[.95,0],[0,1.01]])  #state transition matrix of sensors...
		self.C = np.identity(2)  #combination matrix of observation values

	def predict(self):
		"""
		#Called first.
		#Predicts estimate and error prediction according to model of the situation
		"""
		#update current state
		self.estimate = self.A * self.previousEstimate
		#update error prediction state
		self.errorPrediction = self.A * self.previousErrorPrediction * np.transpose(self.A)

	def update(self,currentObservation):
		"""
		#Called second.
		#Updates estimate according to combination of observed and prediction.
		#Also updates our learning parameters of gain and errorprediction.
		"""
		#update the current estimate based on the gain
		self.estimate = self.estimate + self.gain*(currentObservation - self.C * self.estimate)
		#update the gain based on results from hte previous attempt at estimating
		invVal = self.C * self.errorPrediction * np.transpose(self.C) + self.sensorNoiseProperty
		self.gain = self.errorPrediction * np.transpose(self.C) * np.linalg.inv(invVal)
		#update error prediction based on our success
		self.errorPrediction = (np.identity(2) - self.gain * self.C)*self.errorPrediction

		#update variables for next round
		self.previousEstimate = self.estimate
		self.previousGain = self.gain
		self.previousErrorPrediction = self.errorPrediction;




class ExtendedKalman():
	def __init__(self,initialObservation):
		self.estimate = initialObservation  #current
		self.previousEstimate = initialObservation   
		self.gain = np.identity(2) #tradeoff of system between estimation and observation
		self.previousGain = np.identity(2)
		self.errorPrediction = np.identity(2) #current estimation of the signal error,... starts as identity
		self.previousErrorPrediction = np.identity(2)
		self.sensorNoiseProperty = np.identity(2)*pow(.1,2)  #variance of sensor noise,
		self.f = lambda x: np.array([math.sin(x[0]),math.cos(x[1])])  #state-transition function
		self.h = lambda x: x  #sensor function ... in our case simply identity

	def predict(self):
		"""
		Called first.
		Predicts estimate and error prediction according to model of the situation
		"""
		#update current state
		self.estimate = self.f(self.previousEstimate)
		#update error prediction state
		jac = nd.Jacobian(self.f)
		jacVal = jac(self.previousEstimate)
		self.errorPrediction = jacVal * self.previousErrorPrediction * np.transpose(jacVal)

	def update(self,currentObservation):
		"""
		Called second.
		Updates estimate according to combination of observed and prediction.
		Also updates our learning parameters of gain and errorprediction.
		"""
		#update the current estimate based on the gain
		self.estimate = self.estimate + self.gain*(currentObservation - self.h(self.estimate))

		jac = nd.Jacobian(self.h)
		jacVal = jac(self.estimate)

		#update the gain based on results from hte previous attempt at estimating
		invVal = jacVal * self.errorPrediction * np.transpose(jacVal) + self.sensorNoiseProperty
		self.gain = self.errorPrediction * np.transpose(jacVal) * np.linalg.inv(invVal)
		#update error prediction based on our success
		self.errorPrediction = (np.identity(2) - self.gain * jacVal)*self.errorPrediction

		#update variables for next round
		self.previousEstimate = self.estimate[0]
		self.previousGain = self.gain
		self.previousErrorPrediction = self.errorPrediction;



#variables governing the simulation
startTime = 0
numSamples = 100
samplingRate = 1.0 #in hz

#our sensor simulators ... voltmeter and ammeter
voltmeter = sensors.Voltmeter_Linear(0,1)
ammeter = sensors.Ammeter_Linear(0,3)

#run simulation, log results
x_vals = []
volt_vals = []
current_vals = []
r_vals = []
ekfv_vals = []
ekfc_vals = []

voltVal = voltmeter.getData(startTime)
currentVal = ammeter.getData(startTime)
initialReading = np.array([voltVal,currentVal])  #values are column vectors
kf = SimpleKalman(initialReading)
for i in range(numSamples)[1:]:
	currentTime = startTime + float(i)/samplingRate
	voltVal = voltmeter.getData(currentTime)
	currentVal = ammeter.getData(currentTime)
	reading = np.array([voltVal,currentVal])  #values are column vectors

	kf.predict()
	kf.update(reading)
	#store data for plotting
	x_vals.append(currentTime)
	volt_vals.append(voltVal)
	current_vals.append(currentVal)
	voltage_guess = kf.estimate[0][0]
	current_guess = kf.estimate[1][1]
	ekfv_vals.append(voltage_guess)
	ekfc_vals.append(current_guess)


#display results
plt.plot(x_vals,volt_vals)
plt.plot(x_vals,current_vals)
plt.plot(x_vals,ekfv_vals,marker="1")
plt.plot(x_vals,ekfc_vals,marker="1")
plt.show()



