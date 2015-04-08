import sensors
import math
import numpy as np
import matplotlib.pyplot as plt
import numdifftools as nd




class SimpleKalman():
	"""
	Linear implementation of kalman filter
	Initialize with given values for model, then call predict, then update, then get estimate!
	"""
	def __init__(self,initialObservation,std_sensor1,std_sensor2,scaleFactor_sensor1, scaleFactor_sensor2):
		self.estimate = initialObservation  #current estimate, initialized with initial observation
		self.previousEstimate = initialObservation   #previous estimate, initialized with initial observation
		self.gain = np.identity(2) #tradeoff of system between estimation and observation, initialized as identity arbitrarily
		self.previousGain = np.identity(2) #previous gain, also initialized arbitrarily
		self.errorPrediction = np.identity(2) #current estimation of the signal error,... starts as identity arbitrarily
		self.previousErrorPrediction = np.identity(2)  #previous error, also identity
		self.sensorNoiseProperty = np.array([[math.pow(std_sensor1,2),0],[0,math.pow(std_sensor2,2)]])  #variance of sensor noise values (which is std^2)
		self.A = np.array([[scaleFactor_sensor1,0],[0,scaleFactor_sensor2]])  #state transition matrix of sensors...
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

	def getEstimate(self):
		return self.estimate



class ExtendedKalman():
	"""
	Nonlinear Kalman Filter Implementation
	"""
	def __init__(self,initialObservation, numSensors, sensorNoise, stateTransitionFunction, sensorTransferFunction):
		self.numSensors = numSensors
		self.estimate = initialObservation  #current estimate, initialized with first observation
		self.previousEstimate = initialObservation  #previous state's estimate, initialized with first observation
		self.gain = np.identity(numSensors) #tradeoff of system between estimation and observation, initialized arbitrarily at identity
		self.previousGain = np.identity(numSensors)  #previous gain, again arbitarily initialized
		self.errorPrediction = np.identity(numSensors) #current estimation of the signal error,... starts as identity arbitrarily
		self.previousErrorPrediction = np.identity(numSensors)  #previous signal error, again arbitrary initialization
		self.sensorNoiseProperty = sensorNoise #variance of sensor noise
		self.f = stateTransitionFunction #state-transition function, from user input
		self.fJac = nd.Jacobian(self.f) #jacobian of f
		self.h = sensorTransferFunction  #sensor transfer function, from user input
		self.hJac = nd.Jacobian(self.h) #jacobian of h

	def predict(self):
		"""
		Called first.
		Predicts estimate and error prediction according to model of the situation
		"""
		#update current state
		self.estimate = self.f(self.previousEstimate)

		#find current jacobian value
		jacVal = self.fJac(self.previousEstimate)

		#update error prediction state
		self.errorPrediction = np.dot(jacVal , np.dot(self.previousErrorPrediction,np.transpose(jacVal)))

	def update(self,currentObservation):
		"""
		Called second.
		Updates estimate according to combination of observed and prediction.
		Also updates our learning parameters of gain and errorprediction.
		"""
		#update the current estimate based on the gain
		self.estimate = self.estimate + np.dot(self.gain,(currentObservation - self.h(self.estimate)))
		#find current jacobian value
		jacVal = self.hJac(self.estimate)

		#update the gain based on results from hte previous attempt at estimating
		invVal = np.dot(jacVal, np.dot(self.errorPrediction, np.transpose(jacVal))) + self.sensorNoiseProperty
		self.gain = np.dot(self.errorPrediction, np.dot(np.transpose(jacVal) , np.linalg.inv(invVal) ))
		#update error prediction based on our success
		self.errorPrediction = np.dot((np.identity(self.numSensors) - np.dot(self.gain, jacVal)), self.errorPrediction)

		#update variables for next round
		self.previousEstimate = self.estimate
		self.previousGain = self.gain
		self.previousErrorPrediction = self.errorPrediction;

	def getEstimate(self):
		"""
		Simple getter for cleanliness
		"""
		return self.estimate



#variables governing the simulation
numSamples = 100

#our sensor simulators ... voltmeter and ammeter
voltmeter = sensors.Voltmeter(0,3)
ammeter = sensors.Ammeter(0,2)
#and their associated state transfer functions, sensor transfer functions, and noise values
stateTransfer = lambda x: np.array([[math.pow(x[0][0],1.01)],[math.pow(x[1][0],.99)+5]]) 
sensorTransfer = lambda x: x 
sensorNoise = np.array([[math.pow(3,2),0],[0,math.pow(2,2)]])

#result log holders
x_vals = []
volt_vals = []
current_vals = []
r_vals = []
ekfv_vals = []
ekfc_vals = []
ekfr_vals = []

#finally grab initial readings
voltVal = voltmeter.getData()
currentVal = ammeter.getData()
#put them in a column vector
initialReading = np.array([[voltVal],[currentVal]])  #values are column vectors
#and initialize our filter with our initial reading, our 2 sensors, and all of the associated data
kf = ExtendedKalman(initialReading,2,sensorNoise,stateTransfer,sensorTransfer)

#now run the simulation
for i in range(numSamples)[1:]:
	#grab data
	voltVal = voltmeter.getData() 
	currentVal = ammeter.getData()  
	reading = np.array([[voltVal],[currentVal]])  #values are column vectors

	#predict & update
	kf.predict()
	kf.update(reading)

	#grab result for this iteration and figure out a resistance value
	myEstimate = kf.getEstimate()
	voltage_guess = myEstimate[0][0]
	current_guess = myEstimate[1][0]
	current_resistance = voltage_guess / current_guess

	#store data for plotting
	x_vals.append(i)
	volt_vals.append(voltVal)
	current_vals.append(currentVal)
	ekfv_vals.append(voltage_guess)
	ekfc_vals.append(current_guess)
	ekfr_vals.append(current_resistance)

	print "Done with iter",i


#display results
plt.grid(True)
plt.xlabel('Samples')
plt.ylabel('Voltage')
plt.title('EKF Voltage Tracking')
l1 = plt.plot(x_vals,volt_vals,color='blue', label='Raw Voltage Data')
l2 = plt.plot(x_vals,ekfv_vals,marker="1",color='red', label='Filtered Voltage Data')
plt.legend()
plt.savefig("voltage.png")
plt.clf()

plt.xlabel('Samples')
plt.ylabel('Current')
plt.title('EKF Current Tracking')
plt.plot(x_vals,current_vals,color='blue',label="Raw Current Data")
plt.plot(x_vals,ekfc_vals,marker="1",color='red',label="Filtered Current Data")
plt.legend()
plt.savefig("current.png")
plt.clf()

plt.xlabel('Samples')
plt.ylabel('Resistance')
plt.title('EKF Resistance Tracking')
plt.plot(x_vals,ekfr_vals,marker="2", color='red',label="Filtered Resistance Value")
plt.legend()
plt.savefig("resistance.png")




