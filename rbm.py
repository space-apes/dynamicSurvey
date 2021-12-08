"""
NAME: rbm.py
AUTHOR: Brian Reza Smith
INSTITUTION: California State University Dominguez Hills
YEAR: 2021
DESCRIPTION:
	implementation of restricted boltzmann machine for binary vector {0,1} data. 
	Takes in numpy array of binary column vectors.
	Models probability distribution of training set.


"""
from datetime import datetime
import numpy as np
from mathtools import logistic
import random
from math import log
import sys
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
it is usually helpful to initialize the bias of visible unit i to log[pi/(1-pi)] where pi is the proportion of training vectors in which unit i is on. If this is not done, the early stage of learning will us
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

	def __init__(self, _hiddenUnitCount, _logFileObject, _trainingSet, _logLevel, ):
		self.trainingSet = _trainingSet
		self.hiddenUnitCount = _hiddenUnitCount
		self.logFileObject = _logFileObject
		self.logLevel = _logLevel
		
		#determine number of visible units from length of first training sample
		self.visibleUnitCount = len(self.trainingSet[0])

		#initialize hidden x visible weight matrix to 0s	
		self.weights = np.zeros((self.visibleUnitCount, self.hiddenUnitCount))

		#set initial weights for visible x hidden to sample from gaussian with m = 0, sd= .1
		for row in range(self.visibleUnitCount):
			for col in range(self.hiddenUnitCount):
				self.weights[row][col] = random.gauss(0, .1)
		
		
		#initialize initial hidden bias vector to 0s 
		self.hiddenBiases = np.zeros((self.hiddenUnitCount, 1))
		

		#for each visible unit, set initial bias to log odds for training set
		self.visibleBiases = np.zeros((self.visibleUnitCount, 1))
		matrixSumOfTrainingSet = np.zeros((self.visibleUnitCount,1 ))
		for trainingSample in self.trainingSet:
			matrixSumOfTrainingSet = np.add(matrixSumOfTrainingSet, trainingSample)
		
		numberOfSamples = len(self.trainingSet)

		for i in range(self.visibleUnitCount):
			probSuccess = matrixSumOfTrainingSet[i][0] / numberOfSamples
			probFail = (1-(probSuccess))

			#in 0 cases, avoid breaking logit by setting to very small value
			if probSuccess == 0:
				probSuccess = 10**-5
			if probFail ==0:
				probFail = 10**-5

			self.visibleBiases[i] = log( probSuccess / probFail)
			#self.visibleBiases[i] = log(matrixSumOfTrainingSet[i] / (numberOfSamples - matrixSumOfTrainingSet[i]))
		
		if self.logLevel =='high':
		 	self.printInternals('AFTER INITIALIZATION')

		
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

	
		
	"""
		here we can see stochastic nature of rbm. 
		even with high probability of activation, still some chance it won't fire.
		This introduces noise during training. 
	"""
	def expectedHGivenXVector(self, hIndex, xv):
		probH = self.probHGivenXVector(hIndex,xv)
		random.seed()
		randUniform = random.random()
		if randUniform <= probH:
			return 1 
		else:
			return 0
		#return round(self.probHGivenXVector(hIndex, xv))


	def probXGivenHVector(self, vIndex, hv):
		"""
			@param int vIndex : index of visible node for which we wish to calculate activation probability
			@param colVector hv : binary hidden value activations from which we wish to calculate activation of particular visible node
			
			@return float probability : activation probability
		"""
		#linear combination on hv
		sumOfVectorWeightProducts = 0
		for i in range(len(hv)):
			sumOfVectorWeightProducts += hv[i][0]*self.weights[vIndex][i]
		return logistic(self.visibleBiases[vIndex][0] + sumOfVectorWeightProducts)
	
	def expectedXGivenHVector(self, vIndex, hv):
		"""
			Notice comparison to random samples from uniform distribution. 
			Here we can see stochastic nature of RBM: even with strong collective push from neighboring weights/node states, target may not fire. 

			@param int vIndex : index of visible node for which we wish to calculate state
			@param colVector hv : binary hidden value activations from which we wish to calculate particular visible node state
			
			@return int activation : 0 or 1 value for visible node state
		"""
		probX = self.probXGivenHVector(vIndex,hv)
		random.seed()
		randUniform = random.random()
		if randUniform <= probX:
			return 1 
		else:
			return 0
		#return round(self.probXGivenHVector(vIndex, hv))
	

	def expectedXVectorGivenHVector(self, hv):
		"""
			@param colVector hv : binary hidden value vector from which to generate an entire visible layer activation vector

			@return colVector generatedVisibleLayer : binary visible layer vector from given hidden layer vector
		"""
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
	
	
	def train(self,learningRate, gibbsIterations, timesThroughTrainingSet):
		"""
			one stop shop for batch training of RBM. set all training parameters here. 

			@param float learningRate : weight applied to incremental adjustments made to learned values (weights, biases)
			@param int gibbsIterations : number of times to chain generating visible and hidden vectors
			@param int timesThroughTrainingSet : number of cycles through the training set to update learned values for each training sample

			@return None
		"""
		for trainingSetIterations in range(timesThroughTrainingSet):

			#metric for evaluating reconstruction error
			sumAcrossSamplesForPercentGeneratedCorrectly = 0

			#for all samples in training set
			for i in range(len(self.trainingSet)):	

				#save visible sample from training set, and hidden vector generated from training sample
				hiddenFromTrainingVisible = self.expectedHVectorGivenXVector(self.trainingSet[i])
				
				#initialize fabricated visible and hidden vectors to be generated using gibbs sampling 
				gibbsVisible = np.copy(self.trainingSet[i])
				hiddenFromGibbsVisible = np.zeros((self.visibleUnitCount))

				#do gibbs sampling to set values for fabricated visible and hidden samples
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
				
				#for each sample, add % of visible units reconstructed correctly to sum. If model has been trained, this should approach 100%
				sumAcrossSamplesForPercentGeneratedCorrectly += self.percentOfOverlapBetweenVectors(self.trainingSet[i], gibbsVisible)

			#avg trainingVisible == generatedVisible for trainingVisible->hidden->generatedVisible
			#across all training set items
			self.logFileObject.write(f"RBM.train:: {learningRate} {gibbsIterations} {trainingSetIterations} {sumAcrossSamplesForPercentGeneratedCorrectly / len(self.trainingSet)}\n")
		if self.logLevel == 'high':
			self.printInternals('AFTER TRAINING')
