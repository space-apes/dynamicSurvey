import numpy as np
from random import randint
from mathtools import binVectorHammingDistance, isBinaryColVector, isBinaryColVectorList

class Kmodes:
	distanceFunction = None
	trainingSet = None

	def __init__(self, _distanceFunction, _trainingSet):
		self.distanceFunction = _distanceFunction
		self.trainingSet = _trainingSet

	def modeOfBinVectorList(self, listOfVectors):
		firstShape = np.shape(listOfVectors[0])
		numberOfVectors = len(listOfVectors)

		if (not isBinaryColVectorList(listOfVectors)):
			print("Kmodes.modeOfBinVector:: non binary vector")
			return

		vectorCount = np.zeros((np.shape(listOfVectors[0])))
		for vector in listOfVectors:
			if (np.shape(vector) != firstShape):
				print("Kmodes.modeOfBinVector:: shape of vector does not match others")
				return
			vectorCount = np.add(vectorCount, vector)
		return np.reshape(list( map(lambda x: round( x[0]/numberOfVectors ), vectorCount)), (np.shape(listOfVectors[0])))
	
	def avgHammingDistanceToTargetVector(self, listOfVectors, targetVector):
		totalHammingDistances = 0
		for currentVector in listOfVectors:
			totalHammingDistances += binVectorHammingDistance(currentVector, targetVector)
		return totalHammingDistances / len(listOfVectors)
	
"""		
	
	def kClustersOfIndexes(k):
		indexclusters = []
		centroids = []
		trainingSetSet = np.copy(self.trainingSet)
		for cluster in range(k):
			indexClusters.append([])
			randIndex = randint(0, len(trainingSetSet)-1)
			centroids.append(trainingSetSet[randIndex])
			trainingSetSet.remove(randIndex)
		
		iterationsBeforeConvergence = 0
		converged = True

		while not converged:a
"""
