from datetime import datetime
import numpy as np
from mathtools import logistic
from random import gauss
from math import log

"""
NUMBER OF HIDDEN UNITS:
how many bits it would take to describe each data-vector if you were using a good
model (i.e. estimate the typical negative log2 probability of a datavector under a good model). Then
multiply that estimate by the number of training cases and use a number of parameters that is about
an order of magnitude smaller. If you are using a sparsity target that is very small, you may be able
to use more hidden units. If the training cases are highly redundant, as they typically will be for very
big training sets, you need to use fewer parameters.


INITIAL VALUES OF WEIGHTS:
The weights are typically initialized to small random values chosen from a zero-mean Gaussian with
a standard deviation of about 0.01.

RANDOM VALUES FOR BIASEs: 
it is usually helpful to initialize the bias of visible unit i to log[pi/(1−pi)] where pi is the proportion of training vectors in which unit i is on. If this is not done, the early stage of learning will us
the hidden units to make i turn on with a probability of approximately pi.

Set initial value of hidden unit biases to 0. 
"""

class RBM:
	visibleUnitCount = 0 
	hiddenUnitCount = 0
	trainingSet = []
	visibleBiases = []
	hiddenBiases = []
	#rows are visible, cols are hidden
	weights = []
	logFileObject = None

	def __init__(self, _learningRate, _hiddenUnitCount, _logFileObject, _trainingSet):
		self.trainingSet = _trainingSet
		self.hiddenUnitCount = _hiddenUnitCount
		self.logFileObject = _logFileObject
		self.logInternals("set initial parameters")

		
		#determine number of visible units from length of first training sample
		self.visibleUnitCount = len(self.trainingSet[0])
		self.printInternals("added visible unit count")

		#initialize hidden x visible weight matrix to 0s	
		self.weights = np.zeros((self.visibleUnitCount, self.hiddenUnitCount))

		#set initial weights for visible x hidden to sample from gaussian with m = 0, sd= .1
		for row in range(self.visibleUnitCount):
			for col in range(self.hiddenUnitCount):
				self.weights[row][col] = gauss(0, .1)
		
		self.logInternals("set random values for weight matrix")
		
		#initialize initial hidden bias vector to 0s 
		self.hiddenBiases = np.zeros((self.hiddenUnitCount))
		

		#for each visible unit, set initial bias to log odds for training set
		self.visibleBiases = np.zeros((self.visibleUnitCount))
		matrixSumOfTrainingSet = np.zeros((1, self.visibleUnitCount))
		for trainingSample in self.trainingSet:
			matrixSumOfTrainingSet = np.add(matrixSumOfTrainingSet, trainingSample)
		
		numberOfSamples = len(self.trainingSet)

		#print(f"matrixSumOfTrainingSet: {matrixSumOfTrainingSet}, numberOfSamples: {numberOfSamples}")
		for i in range(self.visibleUnitCount):
			self.visibleBiases[i] = log(matrixSumOfTrainingSet[0][i] / (numberOfSamples - matrixSumOfTrainingSet[0][i]))


	
		
	def printInternals(self, _label):
		now = datetime.now()
		timeStamp = now.strftime("%d/%m/%Y %H:%M:%S")
		message = f"{timeStamp} {_label}\nvisibleUnitCount: {self.visibleUnitCount} hiddenUnitCount: {self.hiddenUnitCount}\n------------------------------------------------\n"
		print(message)

	def logInternals(self, _label):
		now = datetime.now()
		timeStamp = now.strftime("%d/%m/%Y %H:%M:%S")
		message = f"{timeStamp} {_label}\nvisibleUnitCount: {self.visibleUnitCount} hiddenUnitCount: {self.hiddenUnitCount}\n------------------------------------------------\n"
		self.logFileObject.write(message)

	
	
	def probHGivenXVector(self, hIndex, xv):
		#linear combination on xv
		sumOfVectorWeightProducts = 0
		for i in range(len(xv)):
			sumOfVectorWeightProducts += xv[i]*self.weights[i][hIndex]
		return logistic(self.hiddenBiases[hIndex] + sumOfVectorWeightProducts)

	
	def probXGivenHVector(self, vIndex, hv):
		#linear combination on hv
		sumOfVectorWeightProducts = 0
		for i in range(len(hv)):
			sumOfVectorWeightProducts += hv[i]*self.weights[vIndex][i]
		return logistic(self.visibleBiases[vIndex] + sumOfVectorWeightProducts)
		
	"""
		do i need to worry about comparing to uniform bernoulli?

		does rounding work?
	"""
	def expectedHGivenXVector(self, hIndex, xv):
		return round(self.probHGivenXVector(hIndex, xv))

	
	"""
		do i need to worry about comparing to uniform bernoulli?
		
		does rounding work?
	"""
	def expectedXGivenHVector(self, vIndex, hv):
		return round(self.probXGivenHVector(vIndex, hv))
	

	def expectedXVectorGivenHVector(self, hv):
		tempXVector = np.zeros((self.visibleUnitCount))
		for i in range(self.visibleUnitCount):
			tempXVector[i] = self.expectedXGivenHVector(i, hv)
		return tempXVector


	def expectedHVectorGivenXVector(self, xv):
		tempHVector = np.zeros((self.hiddenUnitCount))
		for i in range(self.hiddenUnitCount):
			tempHVector[i] = self.expectedHGivenXVector(i, xv)
		return tempHVector
	
	"""
	initialize starting weights and biases
	go through each training set item
		pick a random set of hidden, visible, and weight elements that matches training set
			from set of ALL configurations that include training element nodes
				figure out conditional probability for each hidden node given visible config
			pick configuration with training visibles, and tack on conditional hiddens
			increase weights for all those weights/biases by learning rate
		pick a random set of hidden,visible, and weight elements (doesnt matter if matches training)
			start with chosen configuration whose weights were increased
			decrease weights for all elements of that set
	do all that stuff multiple times

	"""	
	
	def train(self,learningRate, gibbsIterations):
		for i in range(len(self.trainingSet)):
			tempVisible = np.copy(self.trainingSet[i])
			tempHidden = np.zeros((self.hiddenUnitCount))
			for gibbsIter in range(gibbsIterations):
				tempHidden = self.expectedHVectorGivenXVector(tempVisible)
				tempVisible = self.expectedXVectorGivenHVector(tempHidden)
			print(f"after {gibbsIterations} gibbs iterations: ")
			print(f"\t trainingXVector: {self.trainingSet[i]}")
			print(f"\t generatedXVector: {tempVisible}")
			print(f"\t startingHVector: {np.zeros((self.hiddenUnitCount))}")
			print(f"\t generatedHVector: {tempHidden}")
			print("-" * 25)


	"""
	
	#energyOfConfiguration(visibleBiases, weights, hiddenBiases)
	
			
	#def probXGivenHVector(xIndex, hv): 
	
	#def probHVectorGivenXVector?
		probability of multiple independent events is product of each one
	#def probXVectorGivenHVector?
		probability of multiple independent events is product of each one

	"""	
