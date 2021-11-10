from datetime import datetime
import numpy as np
from mathtools import logistic
from random import gauss
from math import log

"""
NUMPY BROADCASTING : 
However, NumPy also has a concept of broadcasting and one of the rules of broadcasting is that 
extra axes will be automatically added to any array on the left-hand side of its shape whenever an operation requires it. 
So, a 1-dimensional NumPy array of shape (5,) can broadcast to a 2-dimensional array of shape (1,5) 
(or 3-dimensional array of shape (1,1,5), etc).
https://stats.stackexchange.com/questions/284995/are-1-dimensional-numpy-arrays-equivalent-to-vectors


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



WHAT ABOUT LEARNING RATE?!



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

	def __init__(self, _hiddenUnitCount, _logFileObject, _trainingSet):
		self.trainingSet = _trainingSet
		self.hiddenUnitCount = _hiddenUnitCount
		self.logFileObject = _logFileObject
		self.logInternals("set initial parameters")

		
		#determine number of visible units from length of first training sample
		self.visibleUnitCount = len(self.trainingSet[0])

		#initialize hidden x visible weight matrix to 0s	
		self.weights = np.zeros((self.visibleUnitCount, self.hiddenUnitCount))

		#set initial weights for visible x hidden to sample from gaussian with m = 0, sd= .1
		for row in range(self.visibleUnitCount):
			for col in range(self.hiddenUnitCount):
				self.weights[row][col] = gauss(0, .1)
		
		
		#initialize initial hidden bias vector to 0s 
		self.hiddenBiases = np.zeros((self.hiddenUnitCount, 1))
		

		#for each visible unit, set initial bias to log odds for training set
		self.visibleBiases = np.zeros((self.visibleUnitCount, 1))
		matrixSumOfTrainingSet = np.zeros((self.visibleUnitCount,1 ))
		for trainingSample in self.trainingSet:
			matrixSumOfTrainingSet = np.add(matrixSumOfTrainingSet, trainingSample)
		
		numberOfSamples = len(self.trainingSet)

		print(f"matrixSumOfTrainingSet: {matrixSumOfTrainingSet}, numberOfSamples: {numberOfSamples}")

		for i in range(self.visibleUnitCount):
			self.visibleBiases[i] = log(matrixSumOfTrainingSet[i] / (numberOfSamples - matrixSumOfTrainingSet[i]))

		self.logInternals("END OF CONSTRUCTOR")	

		
	def printInternals(self, _label):
		now = datetime.now()
		timeStamp = now.strftime("%d/%m/%Y %H:%M:%S")
		message = f"{timeStamp} {_label}\ntrainingSet shape: {np.shape(self.trainingSet)}, visibleBiases shape: {np.shape(self.visibleBiases)}, hiddenBiasesShape: {np.shape(self.hiddenBiases)}\nhiddenBiases:\n{self.hiddenBiases}\nvisibleBiases:\n{self.visibleBiases}\nweights:\n{self.weights}\n------------------------------------------------\n"
		print(message)

	def logInternals(self, _label):
		now = datetime.now()
		timeStamp = now.strftime("%d/%m/%Y %H:%M:%S")
		message = f"{timeStamp} {_label}\ntrainingSet shape: {np.shape(self.trainingSet)}, visibleBiases shape: {np.shape(self.visibleBiases)}, hiddenBiasesShape: {np.shape(self.hiddenBiases)}\nhiddenBiases:\n{self.hiddenBiases}\nvisibleBiases:\n{self.visibleBiases}\nweights:\n{self.weights}\n------------------------------------------------\n"
		self.logFileObject.write(message)

	
	
	def probHGivenXVector(self, hIndex, xv):
		#linear combination on xv
		sumOfVectorWeightProducts = 0
		for i in range(len(xv)):
			sumOfVectorWeightProducts += xv[i][0]*self.weights[i][hIndex]
		return logistic(self.hiddenBiases[hIndex][0] + sumOfVectorWeightProducts)

	
	def probXGivenHVector(self, vIndex, hv):
		#linear combination on hv
		sumOfVectorWeightProducts = 0
		for i in range(len(hv)):
			sumOfVectorWeightProducts += hv[i][0]*self.weights[vIndex][i]
		return logistic(self.visibleBiases[vIndex][0] + sumOfVectorWeightProducts)
		
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
		tempXVector = np.zeros((self.visibleUnitCount,1))
		for i in range(self.visibleUnitCount):
			tempXVector[i] = self.expectedXGivenHVector(i, hv)
		return np.array(tempXVector)


	def expectedHVectorGivenXVector(self, xv):
		tempHVector = np.zeros((self.hiddenUnitCount,1))
		for i in range(self.hiddenUnitCount):
			tempHVector[i] = self.expectedHGivenXVector(i, xv)
		return np.array(tempHVector)

	def percentOfOverlapBetweenVectors(self,v1, v2):
		if np.shape(v1) != np.shape(v2):
			print("percentOverlapBetweenVectors:: vector shapes do not match")
			return -1
		matchCount = 0 
		for i in range(len(v1)):
			if v1[i][0] == v2[i][0]:
				matchCount+=1
		return (matchCount / len(v1))*100
	
	"""
	initialize starting weights and biases
	for each training set item
		tempVisible = trainingSetItem
		for gibbsIterations cycles
			generate hidden vector using tempVisible
			generate new visible vector from generated hidden

		Weights = Weights + learningRate * ( (trainingVisible * hiddenFromTrainingVisible) - (gibbsVisible*hiddenFromGibbsVisible))
		hiddenBiases = hiddenBiases + learningRate * (hiddenFromTrainingVisible - hiddenFromGibbsVisible) 
		visibleBiases = visibleBiases + learningRate * (trainingVisible - gibbsVisible)
		

	"""	
	
	def train(self,learningRate, gibbsIterations, timesThroughTrainingSet):
		self.logInternals("BEGIN TRAINING")
		for iters in range(timesThroughTrainingSet):
			sumAcrossSamplesForPercentGeneratedCorrectly = 0
			for i in range(len(self.trainingSet)):	
				hiddenFromTrainingVisible = self.expectedHVectorGivenXVector(self.trainingSet[i])
				gibbsVisible = np.copy(self.trainingSet[i])
				hiddenFromGibbsVisible = np.zeros((self.visibleUnitCount))
				for gibbsIter in range(gibbsIterations):
					hiddenFromGibbsVisible = self.expectedHVectorGivenXVector(gibbsVisible)
					gibbsVisible = self.expectedXVectorGivenHVector(hiddenFromGibbsVisible)
			
				
				#weights += learningRate (hiddenFromTrainingVisible*trainingVisible) - (hiddenFromGibbsVisible * gibbsVisible)
				posPhase = np.matmul(self.trainingSet[i], np.reshape(hiddenFromTrainingVisible, (1, self.hiddenUnitCount))) 
				negPhase = np.matmul(gibbsVisible, np.reshape(hiddenFromGibbsVisible, (1, self.hiddenUnitCount))) 
				self.weights = np.add(self.weights, (learningRate * np.subtract(posPhase, negPhase)))
				
				#hidden += learningRate (hiddenFromTrainingVisible - hiddenFromGibbsVisible)
				self.hiddenBiases = np.add(self.hiddenBiases, (learningRate * (np.subtract(hiddenFromTrainingVisible, self.expectedHVectorGivenXVector(gibbsVisible)))))
				
				#visible += learningRate(trainingVisible - gibbsVisible)
				self.visibleBiases = np.add(self.visibleBiases, (learningRate * (np.subtract(self.trainingSet[i], gibbsVisible))))

				sumAcrossSamplesForPercentGeneratedCorrectly += self.percentOfOverlapBetweenVectors(self.trainingSet[i], gibbsVisible)

			print(f"ITERATION: {iters}: avg% generatedCorrectly: {sumAcrossSamplesForPercentGeneratedCorrectly / len(self.trainingSet)}")
		self.logInternals("END TRAINING")
