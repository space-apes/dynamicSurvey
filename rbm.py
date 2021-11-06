from datetime import datetime
import numpy as np
from mathtools import logistic
from random import gauss

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
	learningRate = 0
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
		self.learningRate = _learningRate
		self.hiddenUnitCount = _hiddenUnitCount
		self.logFileObject = _logFileObject
		self.logInternals("set initial parameters")

		
		#determine number of visible units from length of first training sample
		self.visibleUnitCount = len(self.trainingSet[0])
		self.printInternals("added visible unit count")

		#initialize hidden x visible weight matrix to 0s	
		self.weights = np.zeros((self.visibleUnitCount, self.hiddenUnitCount))

		#set initial weights for visible x hidden to gaussian with m = 0, sd= .1
		for row in range(self.visibleUnitCount):
			for col in range(self.hiddenUnitCount):
				self.weights[row][col] = gauss(0, .1)
		
		self.logInternals("set random values for weight matrix")
		
		#initialize hidden bias vector to 0s 
		self.hiddenBiases = np.zeroes(1, self.hiddenUnitCount)
		
		#initialize visible bias vector to 0s
		self.visibleBiases = np.zeroes(1, self.visibleUnitCount)

		#for each unit, set initial bias to log odds of activation for training set
			

	def printInternals(self, _label):
		now = datetime.now()
		timeStamp = now.strftime("%d/%m/%Y %H:%M:%S")
		message = f"{timeStamp} {_label}\nlr: {self.learningRate}, visibleUnitCount: {self.visibleUnitCount} hiddenUnitCount: {self.hiddenUnitCount}\n------------------------------------------------\n"
		print(message)

	def logInternals(self, _label):
		now = datetime.now()
		timeStamp = now.strftime("%d/%m/%Y %H:%M:%S")
		message = f"{timeStamp} {_label}\nlr: {self.learningRate}, visibleUnitCount: {self.visibleUnitCount} hiddenUnitCount: {self.hiddenUnitCount}\n------------------------------------------------\n"
		self.logFileObject.write(message)

	
	
	def probHgivenXVector(hIndex, xv):
		#linear combination on xv
		sumOfVectorWeightProducts = 0
		for i in range(len(xv)):
			sumOfVectorWeightProducts += xv[i]*weights[i][hIndex]
		return logistic(hiddenBiases[hIndex] + sumOfVectorWeightProducts)

	
	def probXgivenHVector(vIndex, hv):
		#linear combination on hv
		sumOfVectorWeightProducts = 0
		for i in range(len(hv)):
			sumOfVectorWeightProducts += hv[i]*weights[vIndex][i]
		return logistic(visibleBiases[vIndex] + sumOfVectorWeightProducts)

	
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
	
	#def train(trainingSet):
		#initialize probabilityDistributionArray
		#"The weights are typically initialized to small random values chosen from a zero-mean Gaussian with a standard deviation of about 0.01."

	
	"""
	
	#energyOfConfiguration(visibleBiases, weights, hiddenBiases)
	
			
	#def probXGivenHVector(xIndex, hv): 
	
	#def probHVectorGivenXVector?
		probability of multiple independent events is product of each one
	#def probXVectorGivenHVector?
		probability of multiple independent events is product of each one

	"""	
